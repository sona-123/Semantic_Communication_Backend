from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import os
import numba
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load T5 model
tok = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device).eval()

print("Warming up T5 model...")
_ = model.generate(**tok("semantic", return_tensors="pt").to(device))
print("Model is ready!")

MODEL_DIR = "./models"

# Quantization functions
def q8(x):
    lo, hi = x.min(), x.max()
    q = np.round((x - lo) * 255 / (hi - lo)).astype(np.uint8)
    return q, (lo, hi)

def dq8(q, lohi):
    lo, hi = lohi
    return q.astype(np.float32) * (hi - lo) / 255 + lo
def cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))  # small epsilon to avoid division by zero
# Optimized FEC components with Numba
POLY = (0b1011011, 0b1111001)  # Octal 133(8) = 0b1011011, 171(8) = 0b1111001
K = 7
NS = 1 << (K - 1)

@numba.njit(nogil=True)
def conv_encode_numba(bits):
    state = 0
    out = np.empty(len(bits) * 2, dtype=np.uint8)
    idx = 0
    for bit in bits:
        state = ((state << 1) | bit) & (NS - 1)
        for g in POLY:
            x = state & g
            parity = 0
            while x:
                parity ^= x & 1
                x >>= 1
            out[idx] = parity
            idx += 1
    return out


@numba.njit(nogil=True)
def viterbi_decode_numba(rx_bits):
    n_bits = len(rx_bits) // 2
    INF = 1e9
    pm = np.full(NS, INF, dtype=np.int32)
    pm[0] = 0
    surv = np.zeros((n_bits, NS), dtype=np.int32)
    
    # Precompute branch metrics
    branch = np.zeros((NS, 2), dtype=np.uint8)
    for s in range(NS):
        for u in (0, 1):
            ns = ((s << 1) | u) & (NS - 1)
            coded = [0, 0]
            for i, g in enumerate(POLY):
                x = ns & g
                parity = 0
                while x:
                    parity ^= x & 1
                    x >>= 1
                coded[i] = parity
            branch[s, u] = (coded[0] << 1) | coded[1]
    
    for i in range(n_bits):
        r_sym = (rx_bits[2*i] << 1) | rx_bits[2*i + 1]
        pm_next = np.full(NS, INF, dtype=np.int32)
        for s in range(NS):
            if pm[s] >= INF:
                continue
            for u in (0, 1):
                ns = ((s << 1) | u) & (NS - 1)
                exp_sym = branch[s, u]
                
                xor = r_sym ^ exp_sym
                dist = 0
                while xor:
                    dist += xor & 1
                    xor >>= 1
                
                metric = pm[s] + dist
                if metric < pm_next[ns]:
                    pm_next[ns] = metric
                    surv[i, ns] = (u << 6) | s
    
    # Traceback
    state = np.argmin(pm)
    decoded = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits-1, -1, -1):
        entry = surv[i, state]
        decoded[i] = (entry >> 6) & 1
        state = entry & (NS - 1)
    return decoded


# Optimized BPSK functions
def bpsk_modulate(bit_array):
    return np.where(bit_array == 1, 1.0+0j, -1.0+0j).astype(np.complex64)

def rayleigh_mmse(sig, snr_db):
    snr_lin = 10 ** (snr_db / 10)
    σ = 1 / np.sqrt(2 * snr_lin)
    h = (np.random.randn(*sig.shape) + 1j*np.random.randn(*sig.shape)).astype(np.complex64) / np.sqrt(2)
    n = (np.random.randn(*sig.shape) + 1j*np.random.randn(*sig.shape)).astype(np.complex64) * σ
    r = h * sig + n
    return (np.conjugate(h) / (np.abs(h)**2 + 1/snr_lin)) * r

def bpsk_demod_hard(sym):
    return (sym.real > 0).astype(np.uint8)

# Denoiser model
class Denoiser(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        return self.net(x)

def load_denoiser_for_snr(snr_db):
    den = Denoiser().to(device)
    path = os.path.join(MODEL_DIR, f"t5_denoiser_snr_{snr_db}.pth")
    den.load_state_dict(torch.load(path, map_location=device))
    den.eval()
    return den

def get_hidden(sentence):
    enc = tok(sentence, return_tensors="pt").to(device)
    return model.encoder(**enc).last_hidden_state.detach().cpu().numpy(), enc

# API endpoints
@app.route("/")
def index():
    return "✅ Semantic Communication API is running!"

@app.route('/embedding', methods=['POST'])
def get_embedding():
    data = request.json
    hs, _ = get_hidden(data["text"])
    return jsonify(embedding=hs.tolist())

@app.route('/noisy_embedding', methods=['POST'])
def get_noisy_embedding():
    data = request.json
    sentence = data["text"]
    snr = int(data["snr"])
    
    hs, _ = get_hidden(sentence)
    flat = hs.flatten()
    q, lohi = q8(flat)
    
    bit_array = np.unpackbits(q).astype(np.uint8)
    fec_bits = conv_encode_numba(bit_array)
    tx = bpsk_modulate(fec_bits)
    eq = rayleigh_mmse(tx, snr)
    rx_bits = bpsk_demod_hard(eq)
    decoded_bits = viterbi_decode_numba(rx_bits)
    rx_bytes = np.packbits(decoded_bits)
    H_noisy = dq8(rx_bytes, lohi).reshape(hs.shape)
    
    return jsonify(noisy_embedding=H_noisy.tolist())

@app.route('/reconstructed_embedding', methods=['POST'])
def get_reconstructed_embedding():
    data = request.json
    den = load_denoiser_for_snr(int(data["snr"]))
    H_t = torch.tensor(data["noisy_embedding"], dtype=torch.float32, device=device)
    H_den = den(H_t).detach().cpu().numpy()
    return jsonify(reconstructed_embedding=H_den.tolist())

@app.route('/decode', methods=['POST'])
def decode():
    data = request.json
    emb = np.squeeze(np.array(data["reconstructed_embedding"]))
    if emb.ndim == 2:
        emb = emb[np.newaxis]
    elif emb.ndim != 3:
        return jsonify(error=f"Invalid embedding shape {emb.shape}"), 400
    hs_t = torch.tensor(emb, dtype=torch.float32, device=device)
    enc_out = BaseModelOutput(last_hidden_state=hs_t)
    out_ids = model.generate(
        encoder_outputs=enc_out,
        decoder_input_ids=tok(data["original_text"], return_tensors="pt").to(device)["input_ids"][:, :1],
        max_length=emb.shape[1] + 10,
        num_beams=5,
        early_stopping=True
    )
    return jsonify(decoded_text=tok.decode(out_ids[0], skip_special_tokens=True))
@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.json
    if "original_embedding" not in data or "reconstructed_embedding" not in data:
        return jsonify(error="Both 'original_embedding' and 'reconstructed_embedding' must be provided."), 400

    original = np.array(data["original_embedding"])
    reconstructed = np.array(data["reconstructed_embedding"])
    sim = cosine_similarity(original, reconstructed)

    return jsonify(similarity=sim)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)