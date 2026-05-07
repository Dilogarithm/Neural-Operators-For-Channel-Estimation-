import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from neuralop.models import FNO


# =========================
# Model (UNCHANGED LOGIC)
# =========================

class SFT_FNO(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()

        self.fno = FNO(
            n_modes=(8, 8, 8),
            hidden_channels=hidden_channels,
            in_channels=4,
            out_channels=2,
        )

        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, R):
        # R shape: (B, 2, NR, NT, K, 2)

        R_prev = R[:, 0]
        R_current = R[:, 1]

        R_prev = R_prev.permute(0, 4, 1, 2, 3)
        R_current = R_current.permute(0, 4, 1, 2, 3)

        x = torch.cat([R_prev, R_current], dim=1)

        correction = self.fno(x)

        # residual refinement (paper style)
        H_est = R_current + self.beta * correction

        return H_est.permute(0, 2, 3, 4, 1)


# =========================
# Beamforming
# =========================

def dft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    W = torch.exp(-2j * math.pi * k * n / N)
    return W / math.sqrt(N)


def generate_beams(NT, NR, MT, MR):
    F = dft_matrix(NT)[:, :MT]
    W = dft_matrix(NR)[:, :MR]
    return F, W


def compute_GL_GR(F, W):
    GL = torch.linalg.inv(W @ W.conj().T) @ W
    GR = F.conj().T @ torch.linalg.inv(F @ F.conj().T)
    return GL, GR


# =========================
# Measurement
# =========================

def generate_measurement(H, F, W, noise_std):
    B, NR, NT, K, _ = H.shape
    Hc = H[..., 0] + 1j * H[..., 1]

    Y = []
    for k in range(K):
        Hk = Hc[:, :, :, k]
        Yk = W.conj().T @ Hk @ F

        noise = noise_std * (
            torch.randn_like(Yk) + 1j * torch.randn_like(Yk)
        ) / math.sqrt(2)

        Y.append(Yk + noise)

    return torch.stack(Y, dim=-1)


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
# Channel generator
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
# DATASET (FIXED CORE PART)
# =========================

def generate_sample(B, NR, NT, K, rho, noise_std, F, W, GL, GR):
    # generate two correlated channels
    H_prev = generate_physical_channel(B, NR, NT, K)
    innovation = generate_physical_channel(B, NR, NT, K)
    H_curr = rho * H_prev + math.sqrt(1 - rho**2) * innovation

    # measurements
    Y_prev = generate_measurement(H_prev, F, W, noise_std)
    Y_curr = generate_measurement(H_curr, F, W, noise_std)

    R_prev = compute_R(Y_prev, GL, GR)
    R_curr = compute_R(Y_curr, GL, GR)

    # stack for model input
    R = torch.stack([R_prev, R_curr], dim=1)

    return R, H_curr


# =========================
# TRAIN
# =========================

def train_model(model, noise_std, NR, NT, K, rho, device, F, W, GL, GR):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()

    for step in range(500):
        R, H = generate_sample(16, NR, NT, K, rho, noise_std, F, W, GL, GR)

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
# EVAL
# =========================

def nmse(pred, target):
    return ((pred - target) ** 2).sum() / (target ** 2).sum()


def evaluate_model(model, noise_std, NR, NT, K, rho, device, F, W, GL, GR):
    model.eval()

    nmse_model, nmse_ls = 0, 0

    with torch.no_grad():
        for _ in range(300):
            R, H = generate_sample(1, NR, NT, K, rho, noise_std, F, W, GL, GR)

            R = R.to(device)
            H = H.to(device)

            scale = R.abs().mean() + 1e-6
            R = R / scale
            H = H / scale

            pred = model(R)

            H_ls = R[:, 1]  # LS baseline

            nmse_model += nmse(pred, H).item()
            nmse_ls += nmse(H_ls, H).item()

    return nmse_model / 300, nmse_ls / 300


# =========================
# MAIN
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NR, NT, K = 6, 6, 16
    rho = 0.9

    F, W = generate_beams(NT, NR, NT, NR)
    GL, GR = compute_GL_GR(F, W)

    F, W, GL, GR = F.to(device), W.to(device), GL.to(device), GR.to(device)

    snr_db_list = [0, 5, 10, 15, 20]
    noise_list = [10 ** (-snr / 20) for snr in snr_db_list]

    results_model, results_ls = [], []

    for snr_db, noise_std in zip(snr_db_list, noise_list):
        print(f"\n===== SNR = {snr_db} dB =====")

        model = SFT_FNO().to(device)
        model = train_model(model, noise_std, NR, NT, K, rho, device, F, W, GL, GR)

        nmse_m, nmse_l = evaluate_model(model, noise_std, NR, NT, K, rho, device, F, W, GL, GR)

        print(f"Model NMSE: {nmse_m:.6f}")
        print(f"LS NMSE:    {nmse_l:.6f}")

        results_model.append(10 * math.log10(nmse_m))
        results_ls.append(10 * math.log10(nmse_l))

    plt.plot(snr_db_list, results_model, 'o-', label="SFT-FNO")
    plt.plot(snr_db_list, results_ls, 'x--', label="LS")
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()