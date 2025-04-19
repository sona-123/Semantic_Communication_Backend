from flask import Flask, request, jsonify
import torch
import numpy as np
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load T5 model
tok = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device).eval()

MODEL_DIR = "./models"

# Quantization
def q8(x): lo, hi = x.min(), x.max(); q = np.round((x - lo) * 255 / (hi - lo)).astype(np.uint8); return q, (lo, hi)
def dq8(q, lohi): lo, hi = lohi; return q.astype(np.float32) * (hi - lo) / 255 + lo

# Conv FEC
POLY = (0o133, 0o171)
K = 7
NS = 1 << (K - 1)

def conv_encode(bitstr):
    state = 0; out = []
    for b in bitstr:
        bit = int(b); state = ((state << 1) | bit) & (NS - 1)
        for g in POLY:
            out.append(str(bin(state & g).count("1") & 1))
    return "".join(out)

def viterbi_decode(rx_bits):
    n_bits = len(rx_bits) // 2
    INF = 1e9
    pm = np.full(NS, INF, dtype=np.int32)
    pm[0] = 0
    surv = np.zeros((n_bits, NS), dtype=np.int8)
    branch = np.zeros((NS, 2), dtype=np.uint8)
    for s in range(NS):
        for u in (0, 1):
            ns = ((s << 1) | u) & (NS - 1)
            coded = [(bin(ns & g).count("1") & 1) for g in POLY]
            branch[s, u] = (coded[0] << 1) | coded[1]

    for i in range(n_bits):
        r_sym = int(rx_bits[2 * i:2 * i + 2], 2)
        pm_next = np.full(NS, INF, dtype=np.int32)
        surv_sym = np.zeros(NS, dtype=np.int8)
        for s in range(NS):
            if pm[s] >= INF: continue
            for u in (0, 1):
                ns = ((s << 1) | u) & (NS - 1)
                exp_sym = branch[s, u]
                dist = ((r_sym >> 1) ^ (exp_sym >> 1)) + ((r_sym & 1) ^ (exp_sym & 1))
                metric = pm[s] + dist
                if metric < pm_next[ns]:
                    pm_next[ns] = metric
                    surv_sym[ns] = (u << 6) | s
        pm = pm_next
        surv[i] = surv_sym

    state = pm.argmin()
    decoded = []
    for i in range(n_bits - 1, -1, -1):
        up = surv[i, state]
        decoded.append(str(up >> 6))
        state = up & (NS - 1)
    return "".join(reversed(decoded))

# BPSK
def bpsk_modulate(bitstr):
    return np.array([1.0 if b == "1" else -1.0 for b in bitstr], dtype=np.complex64)

def rayleigh_mmse(sig, snr_db):
    snr_lin = 10 ** (snr_db / 10)
    σ = 1 / np.sqrt(2 * snr_lin)
    h = (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)).astype(np.complex64) / np.sqrt(2)
    n = (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)).astype(np.complex64) * σ
    r = h * sig + n
    return (np.conjugate(h) / (np.abs(h) ** 2 + 1 / snr_lin)) * r

def bpsk_demod_hard(sym):
    return "".join("1" if x.real > 0 else "0" for x in sym)

# Denoiser model
class Denoiser(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x): return self.net(x)

def load_denoiser_for_snr(snr_db):
    den = Denoiser().to(device)
    path = os.path.join(MODEL_DIR, f"t5_denoiser_snr_{snr_db}.pth")
    den.load_state_dict(torch.load(path, map_location=device))
    den.eval()
    return den

def get_hidden(sentence):
    enc = tok(sentence, return_tensors="pt").to(device)
    return model.encoder(**enc).last_hidden_state.detach().cpu().numpy(), enc

def decode_from_hidden(hs, dec_primer):
    hs_t = torch.tensor(hs, dtype=torch.float32, device=device)
    enc_out = BaseModelOutput(last_hidden_state=hs_t)
    out_ids = model.generate(
        encoder_outputs=enc_out,
        decoder_input_ids=dec_primer,
        max_length=hs.shape[1] + 10,
        num_beams=5,
        early_stopping=True
    )
    return tok.decode(out_ids[0], skip_special_tokens=True)

@app.route("/")
def index():
    return "✅ Semantic Communication API is running!"

@app.route('/embedding', methods=['POST'])
def get_embedding():
    data = request.json
    sentence = data["text"]
    hs, _ = get_hidden(sentence)
    return jsonify(embedding=hs.tolist())

@app.route('/noisy_embedding', methods=['POST'])
def get_noisy_embedding():
    data = request.json
    sentence = data["text"]
    snr = int(data["snr"])
    hs, _ = get_hidden(sentence)
    flat = hs.flatten()
    q, lohi = q8(flat)
    bits = "".join(f"{b:08b}" for b in q)
    fec_bits = conv_encode(bits)
    tx = bpsk_modulate(fec_bits)
    eq = rayleigh_mmse(tx, snr)
    rx_bits = viterbi_decode(bpsk_demod_hard(eq))
    rx_bytes = np.frombuffer(bytes(int(rx_bits[i:i + 8], 2) for i in range(0, len(rx_bits), 8)), dtype=np.uint8)
    H_noisy = dq8(rx_bytes, lohi).reshape(hs.shape)
    return jsonify(noisy_embedding=H_noisy.tolist())

@app.route('/reconstructed_embedding', methods=['POST'])
def get_reconstructed_embedding():
    data = request.json
    noisy_embedding = np.array(data["noisy_embedding"])
    snr = int(data["snr"])
    den = load_denoiser_for_snr(snr)
    H_t = torch.tensor(noisy_embedding, dtype=torch.float32, device=device)
    H_den = den(H_t).unsqueeze(0).detach().cpu().numpy()
    return jsonify(reconstructed_embedding=H_den.tolist())

@app.route('/decode', methods=['POST'])
def decode():
    data = request.json

    # 1️⃣  squeeze away excess 1‑dims → result is either 2‑D or 3‑D
    emb = np.squeeze(np.array(data["reconstructed_embedding"]))

    # 2️⃣  guarantee a batch‑dim
    if emb.ndim == 2:          # [seq, hidden]
        emb = emb[np.newaxis]  # -> [1, seq, hidden]
    elif emb.ndim != 3:        # any other rank is an error
        return jsonify(error=f"Bad embedding shape {emb.shape}"), 400

    hs_t = torch.tensor(emb, dtype=torch.float32, device=device)
    enc_out = BaseModelOutput(last_hidden_state=hs_t)

    # primer token from the original sentence
    enc = tok(data["original_text"], return_tensors="pt").to(device)
    primer = enc["input_ids"][:, :1]

    out_ids = model.generate(
        encoder_outputs=enc_out,
        decoder_input_ids=primer,
        max_length=emb.shape[1] + 10,
        num_beams=5,
        early_stopping=True
    )
    decoded = tok.decode(out_ids[0], skip_special_tokens=True)
    return jsonify(decoded_text=decoded)

if __name__ == '__main__':
    app.run(debug=True)
