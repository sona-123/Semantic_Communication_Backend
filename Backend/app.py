from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from numpy import dot
from numpy.linalg import norm

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once here
tok = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device).eval()

print("Warming up T5 model...")
_ = model.generate(**tok("semantic", return_tensors="pt").to(device))
print("Model is ready!")

MODEL_DIR = "./models"

# Quantization
def q8(x): lo, hi = x.min(), x.max(); q = np.round((x - lo) * 255 / (hi - lo)).astype(np.uint8); return q, (lo, hi)
def dq8(q, lohi): lo, hi = lohi; return q.astype(np.float32) * (hi - lo) / 255 + lo

# BPSK
def bpsk_modulate(bitstr):
    return np.array([1.0 if b == "1" else -1.0 for b in bitstr], dtype=np.complex64)

def rayleigh_mmse(sig, snr_db):
    # Convert SNR from dB to linear scale
    print("Signal recieved (after modulation):", sig)
    snr_lin = 10 ** (snr_db / 10)
    
    # Compute the noise standard deviation based on SNR
    σ = 1 / np.sqrt(2 * snr_lin)
    
    # Rayleigh fading channel (h) and noise (n)
    h = (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)).astype(np.complex64) / np.sqrt(2)
    n = (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)).astype(np.complex64) * σ
    
    # Received signal (signal + noise)
    r = h * sig + n
    
    # MMSE Equalizer - The actual signal recovery step
    equalized_signal = (np.conjugate(h) / (np.abs(h) ** 2 + 1 / snr_lin)) * r
    
    # Optionally, print the received and equalized signals
    print("Received Signal (after Rayleigh noise):", r)
    print("Equalized Signal (after MMSE):", equalized_signal)
    
    # Return the equalized signal
    return equalized_signal


def bpsk_demod_hard(sym):
    return "".join("1" if x.real > 0 else "0" for x in sym)

def cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))  # small epsilon to avoid division by zero

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
    # print("After Quantization,",bits)
    tx = bpsk_modulate(bits)
    # print("After Modulation : ", tx)
    eq = rayleigh_mmse(tx, snr)
    rx_bits = bpsk_demod_hard(eq)
    # print("After demodulation, ", rx_bits)
    rx_bytes = np.frombuffer(bytes(int(rx_bits[i:i + 8], 2) for i in range(0, len(rx_bits), 8)), dtype=np.uint8)
    H_noisy = dq8(rx_bytes, lohi).reshape(hs.shape)
    # print("After dequantization,", H_noisy)
    return jsonify(noisy_embedding=H_noisy.tolist(),
    original_bitstream=bits,           # 👈 original bits before modulation
    noisy_bitstream=rx_bits)

@app.route('/reconstructed_embedding', methods=['POST'])
def get_reconstructed_embedding():
    data = request.json
    noisy_embedding = np.array(data["noisy_embedding"])
    snr = int(data["snr"])
    den = load_denoiser_for_snr(snr)
    H_t = torch.tensor(noisy_embedding, dtype=torch.float32, device=device)  # ❌ no .half()
    H_den = den(H_t).unsqueeze(0).detach().cpu().numpy()
    return jsonify(reconstructed_embedding=H_den.tolist())

@app.route('/decode', methods=['POST'])
def decode():
    data = request.json
    emb = np.squeeze(np.array(data["reconstructed_embedding"]))
    if emb.ndim == 2:
        emb = emb[np.newaxis]
    elif emb.ndim != 3:
        return jsonify(error=f"Bad embedding shape {emb.shape}"), 400
    hs_t = torch.tensor(emb, dtype=torch.float32, device=device)
    enc_out = BaseModelOutput(last_hidden_state=hs_t)
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
@app.route('/similarity', methods=['POST'])
def get_similarity():
    data = request.json
    if "original_embedding" not in data or "reconstructed_embedding" not in data:
        return jsonify(error="Both 'original_embedding' and 'reconstructed_embedding' must be provided."), 400

    original = np.array(data["original_embedding"])
    reconstructed = np.array(data["reconstructed_embedding"])
    sim = cosine_similarity(original, reconstructed)

    return jsonify(similarity=sim)
import io
import base64
from flask import send_file
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

@app.route('/visualize_embeddings', methods=['POST'])
def visualize_embeddings():
    data = request.json
    try:
        original = np.array(data["original_embedding"]).reshape(-1, 512)
        noisy = np.array(data["noisy_embedding"]).reshape(-1, 512)
        reconstructed = np.array(data["reconstructed_embedding"]).reshape(-1, 512)

        # Combine for PCA
        combined = np.vstack([original, noisy, reconstructed])
        reduced = PCA(n_components=2).fit_transform(combined)

        N = original.shape[0]
        orig_2d = reduced[:N]
        noisy_2d = reduced[N:2*N]
        rec_2d = reduced[2*N:]

        # Plotting
        plt.figure(figsize=(10, 6))
        for i in range(N):
            x = [orig_2d[i, 0], noisy_2d[i, 0], rec_2d[i, 0]]
            y = [orig_2d[i, 1], noisy_2d[i, 1], rec_2d[i, 1]]
            plt.plot(x, y, marker='o', linestyle='-', linewidth=1.2)

        plt.title("Embedding Flow: Original → Noisy → Reconstructed")
        plt.xlabel("PCA Dim 1")
        plt.ylabel("PCA Dim 2")
        plt.grid(True)

        # Save to memory buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return send_file(buf, mimetype='image/png')

    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    print("Warming up T5 model...")
    _ = model.generate(**tok("semantic", return_tensors="pt").to(device))
    print("Model is ready!")
    app.run(debug=True)
