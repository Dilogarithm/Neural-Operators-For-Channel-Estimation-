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
            n_modes=(3, 3, 3),
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
# Beamforming + Measurement
# =========================

def dft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    W = torch.exp(-2j * math.pi * k * n / N)
    return W / math.sqrt(N)


def generate_beams(NT, NR, MT, MR):
    F_full = dft_matrix(NT)
    W_full = dft_matrix(NR)

    F = F_full[:, :MT]
    W = W_full[:, :MR]

    return F, W


def compute_GL_GR(F, W):
    GL = torch.linalg.inv(W @ W.conj().T) @ W
    GR = F.conj().T @ torch.linalg.inv(F @ F.conj().T)
    return GL, GR


def generate_measurement(H, F, W, noise_std):
    B, NR, NT, K, _ = H.shape

    H_complex = H[..., 0] + 1j * H[..., 1]

    Y = []

    for k in range(K):
        Hk = H_complex[:, :, :, k]

        Yk = W.conj().T @ Hk @ F

        noise = noise_std * (
            torch.randn_like(Yk) + 1j * torch.randn_like(Yk)
        ) / math.sqrt(2)

        Y.append(Yk + noise)

    Y = torch.stack(Y, dim=-1)

    return Y


def compute_R(Y, GL, GR):
    B, MR, MT, K = Y.shape

    R = []

    for k in range(K):
        Yk = Y[:, :, :, k]
        Rk = GL @ Yk @ GR
        R.append(Rk)

    R = torch.stack(R, dim=-1)

    return torch.stack([R.real, R.imag], dim=-1)


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


# =========================
# Data generation (UPDATED)
# =========================

def generate_data(B, T, NR, NT, K, rho, noise_std, F, W, GL, GR):
    H_t = generate_physical_channel(B, NR, NT, K)

    R_list = []

    for _ in range(T):
        innovation = generate_physical_channel(B, NR, NT, K)
        H_t = rho * H_t + (1 - rho) * innovation

        Y = generate_measurement(H_t, F, W, noise_std)
        R_t = compute_R(Y, GL, GR)

        R_list.append(R_t)

    R = torch.stack(R_list, dim=1)

    return R, H_t


# =========================
# Metrics
# =========================

def nmse(pred, target):
    return ((pred - target) ** 2).sum() / (target ** 2).sum()


def to_db(x):
    return 10 * math.log10(x + 1e-12)


# =========================
# Train
# =========================

def train_model(model, noise_std, NR, NT, K, rho, device, F, W, GL, GR, steps=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()

    for step in range(steps):
        R, H = generate_data(16, 2, NR, NT, K, rho, noise_std, F, W, GL, GR)

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
            print(f"Step {step}, Loss: {loss.item():.6f}")

    return model


# =========================
# Evaluate
# =========================

def evaluate_model(model, noise_std, NR, NT, K, rho, device, F, W, GL, GR):
    model.eval()

    nmse_model = 0
    nmse_avg = 0
    nmse_ls = 0

    with torch.no_grad():
        for _ in range(100):
            R, H = generate_data(1, 2, NR, NT, K, rho, noise_std, F, W, GL, GR)

            R = R.to(device)
            H = H.to(device)

            scale = R.abs().mean() + 1e-6
            R = R / scale
            H = H / scale

            pred = model(R)

            H_ls = R[:, -1]
            H_avg = 0.5 * (R[:, -1] + R[:, -2])

            nmse_model += nmse(pred, H).item()
            nmse_avg += nmse(H_avg, H).item()
            nmse_ls += nmse(H_ls, H).item()

    return nmse_model / 100, nmse_avg / 100, nmse_ls / 100


# =========================
# Main
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NR = 6
    NT = 6
    K = 16
    rho = 0.9

    MT = 6
    MR = 6

    # Beamforming setup
    F, W = generate_beams(NT, NR, MT, MR)
    GL, GR = compute_GL_GR(F, W)

    F = F.to(device)
    W = W.to(device)
    GL = GL.to(device)
    GR = GR.to(device)

    snr_db_list = [0, 5, 10, 15, 20, 25]
    noise_list = [10 ** (-snr / 20) for snr in snr_db_list]

    results_model = []
    results_avg = []
    results_ls = []

    for snr_db, noise_std in zip(snr_db_list, noise_list):
        print(f"\n===== SNR = {snr_db} dB =====")

        model = SFT_FNO().to(device)

        model = train_model(model, noise_std, NR, NT, K, rho, device, F, W, GL, GR)

        nmse_m, nmse_a, nmse_l = evaluate_model(
            model, noise_std, NR, NT, K, rho, device, F, W, GL, GR
        )

        print(f"SFT-FNO NMSE: {nmse_m:.6f}")
        print(f"AVG NMSE:     {nmse_a:.6f}")
        print(f"LS NMSE:      {nmse_l:.6f}")

        results_model.append(to_db(nmse_m))
        results_avg.append(to_db(nmse_a))
        results_ls.append(to_db(nmse_l))

    plt.figure(figsize=(6,4))
    plt.plot(snr_db_list, results_model, 'o-', label="SFT-FNO")
    plt.plot(snr_db_list, results_avg, 's--', label="Averaging")
    plt.plot(snr_db_list, results_ls, 'x--', label="LS")

    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("NMSE vs SNR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
