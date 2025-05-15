import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy import dot
from numpy.linalg import norm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# Initialize the device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load T5 tokenizer and model
tok = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device).eval()

# Define functions for BPSK modulation, Rayleigh fading, and cosine similarity
def bpsk_modulate(bitstr):
    return np.array([1.0 if b == "1" else -1.0 for b in bitstr], dtype=np.complex64)

def rayleigh_mmse(sig, snr_db):
    snr_lin = 10 ** (snr_db / 10)
    σ = 1 / np.sqrt(2 * snr_lin)
    h = (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)).astype(np.complex64) / np.sqrt(2)
    n = (np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)).astype(np.complex64) * σ
    r = h * sig + n
    return (np.conjugate(h) / (np.abs(h) ** 2 + 1 / snr_lin)) * r

def cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))  # small epsilon to avoid division by zero

# Function to get hidden states of the input sentence
def get_hidden(sentence):
    enc = tok(sentence, return_tensors="pt").to(device)
    return model.encoder(**enc).last_hidden_state.detach().cpu().numpy(), enc

# Function for denoising (placeholder for now)
class Denoiser(torch.nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )

    def forward(self, x):
        return self.net(x)

def load_denoiser_for_snr(snr_db):
    den = Denoiser().to(device)
    path = f"./models/t5_denoiser_snr_{snr_db}.pth"
    den.load_state_dict(torch.load(path, map_location=device))
    den.eval()
    return den

# Function to perform Monte Carlo trials
def run_monte_carlo_trials(sentence, snr_values, trials=200):
    similarity_results = {snr: [] for snr in snr_values}
    
    for snr in snr_values:
        for _ in range(trials):
            # Get hidden state for the sentence
            hs, _ = get_hidden(sentence)
            flat = hs.flatten()
            
            # Simulate BPSK modulation and Rayleigh fading
            bits = "".join(f"{b:08b}" for b in np.round((flat - flat.min()) * 255 / (flat.max() - flat.min())).astype(np.uint8))
            tx = bpsk_modulate(bits)
            eq = rayleigh_mmse(tx, snr)
            rx_bits = "".join("1" if x.real > 0 else "0" for x in eq)
            
            # Decode the bits back to an embedding
            rx_bytes = np.frombuffer(bytes(int(rx_bits[i:i + 8], 2) for i in range(0, len(rx_bits), 8)), dtype=np.uint8)
            H_noisy = (rx_bytes.astype(np.float32) * (flat.max() - flat.min()) / 255 + flat.min()).reshape(hs.shape)
            
            # Compute cosine similarity between the original and noisy embeddings
            similarity = cosine_similarity(hs.flatten(), H_noisy.flatten())
            similarity_results[snr].append(similarity)
    
    return similarity_results

# SNR values for Monte Carlo simulation
snr_values = np.linspace(0, 20, 10)  # From 0 dB to 20 dB
sentence = "A primary division for the discussion of clauses is the distinction between independent clauses and dependent clauses.[3] An independent clause can stand alone, i.e. it can constitute a complete sentence by itself. A dependent clause, by contrast, relies on an independent clause's presence to be efficiently utilizable."

# Main function to run the simulation and plotting
def main():
    # Run the Monte Carlo simulations
    similarity_results = run_monte_carlo_trials(sentence, snr_values)
    
    # Calculate average similarity for each SNR
    avg_similarity = {snr: np.mean(similarity_results[snr]) for snr in snr_values}
    
    # Print the results as a table
    print("SNR (dB) | Average Cosine Similarity")
    print("-" * 40)
    for snr, avg_sim in avg_similarity.items():
        print(f"{snr:.2f}   | {avg_sim:.4f}")
    
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(snr_values, list(avg_similarity.values()), marker='o', linestyle='-', color='b')
    plt.title("SNR vs Average Cosine Similarity")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average Cosine Similarity")
    plt.grid(True)
    plt.show()

# Entry point
if __name__ == "__main__":
    main()
