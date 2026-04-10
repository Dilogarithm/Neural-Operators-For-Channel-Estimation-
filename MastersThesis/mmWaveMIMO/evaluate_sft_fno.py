import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from neuralop.models import FNO


# =========================
# Model
# =========================

class SFT_FNO(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()

        self.fno = FNO(
            n_modes=(3, 3, 3), #change later for better results to (12, 12, 12),
            hidden_channels=hidden_channels,
            in_channels=4,
            out_channels=2,
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, R):
        R_current = R[:, -1]
        R_prev = R[:, -2]

        R_current_c = R_current.permute(0, 4, 1, 2, 3)
        R_prev_c = R_prev.permute(0, 4, 1, 2, 3)

        x = torch.cat([R_current_c, R_prev_c], dim=1)

        correction = self.fno(x)

        H_c = R_current_c + self.alpha * correction

        return H_c.permute(0, 2, 3, 4, 1)


# =========================
# Channel Generator
# =========================

def steering_vector(N, angle):
    n = torch.arange(N)
    return torch.exp(1j * math.pi * n * torch.sin(angle))


def generate_physical_channel(B, NR, NT, K, L=3):
    H = torch.zeros(B, NR, NT, K, dtype=torch.cfloat)

    fc = 28e9
    delta_f = 15e3

    for l in range(L):
        alpha = (torch.randn(B) + 1j * torch.randn(B)) * math.exp(-0.5 * l)

        theta = torch.rand(B) * math.pi - math.pi / 2
        phi = torch.rand(B) * math.pi - math.pi / 2
        tau = torch.rand(B) * 200e-9

        for b in range(B):
            a_tx = steering_vector(NT, theta[b])
            a_rx = steering_vector(NR, phi[b])

            for k in range(K):
                f_k = fc + k * delta_f
                phase = torch.exp(-1j * 2 * math.pi * f_k * tau[b])

                H[b, :, :, k] += alpha[b] * torch.outer(a_rx, a_tx.conj()) * phase

    return torch.stack([H.real, H.imag], dim=-1)


def generate_data(B, T, NR, NT, K, rho, noise_std):
    H_t = generate_physical_channel(B, NR, NT, K)

    R_list = []

    for _ in range(T):
        innovation = generate_physical_channel(B, NR, NT, K)
        H_t = rho * H_t + (1 - rho) * innovation

        noise = noise_std * torch.randn_like(H_t)
        R_t = H_t + noise

        R_list.append(R_t)

    R = torch.stack(R_list, dim=1)

    return R, H_t


# =========================
# Metrics
# =========================

def nmse(pred, target):
    return ((pred - target) ** 2).sum() / (target ** 2).sum()


# =========================
# Train one model for given SNR
# =========================

def train_model(model, noise_std, NR, NT, K, rho, device, steps=200): # change to 1500 later for better results
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()

    for step in range(steps):
        R, H = generate_data(16, 2, NR, NT, K, rho, noise_std)

        R = R.to(device)
        H = H.to(device)

        scale = R.abs().mean() + 1e-6
        R = R / scale
        H = H / scale

        pred = model(R)
        loss = ((pred - H) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"[SNR training] Step {step}, Loss: {loss.item():.6f}")

    return model


# =========================
# Evaluate model
# =========================

def evaluate_model(model, noise_std, NR, NT, K, rho, device):
    model.eval()

    total_nmse = 0

    with torch.no_grad():
        for _ in range(100):
            R, H = generate_data(1, 2, NR, NT, K, rho, noise_std)

            R = R.to(device)
            H = H.to(device)

            scale = R.abs().mean() + 1e-6
            R = R / scale
            H = H / scale

            pred = model(R)

            total_nmse += nmse(pred, H).item()

    return total_nmse / 100


# =========================
# Main
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NR = 6
    NT = 6
    K = 16
    rho = 0.9

    # Define SNR in dB
    snr_db_list = [0, 5, 10, 15, 20, 25]

    # Convert SNR → noise_std
    noise_list = [10 ** (-snr / 20) for snr in snr_db_list]

    results = []

    for snr_db, noise_std in zip(snr_db_list, noise_list):
        print(f"\n===== SNR = {snr_db} dB =====")

        model = SFT_FNO().to(device)

        model = train_model(model, noise_std, NR, NT, K, rho, device)

        nmse_val = evaluate_model(model, noise_std, NR, NT, K, rho, device)

        print(f"SNR {snr_db} dB → NMSE: {nmse_val:.6f}")

        results.append(nmse_val)

    # =========================
    # Plot
    # =========================

    plt.figure(figsize=(6,4))

    plt.plot(snr_db_list, results, 'o-', label="SFT-FNO")

    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE")
    plt.title("NMSE vs SNR")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()