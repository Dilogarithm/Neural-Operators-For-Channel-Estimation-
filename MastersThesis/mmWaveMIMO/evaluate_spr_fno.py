import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from neuralop.models import FNO


# =========================
# Model: SPR-FNO
# =========================

N_T = 16
N_R = 16

M_T = 12
M_R = 12



class SPR_FNO(nn.Module):
    def __init__(self, hidden_channels=64, D=4):
        super().__init__()

        self.D = D

        self.fno = FNO(
            n_modes=(6, 6, 8),
            hidden_channels=hidden_channels,
            in_channels=2 * D,
            out_channels=2,
        )

    def forward(self, R):
        # R: (B, D, NR, NT, K, 2)
        B, D, NR, NT, K, _ = R.shape

        # SPR weighting (earlier frames more reliable)
        weights = torch.linspace(1.0, 0.6, D).to(R.device)
        R = R * weights.view(1, D, 1, 1, 1, 1)

        # Prepare FNO input
        R_in = R.permute(0, 1, 5, 2, 3, 4)
        R_in = R_in.reshape(B, 2 * D, NR, NT, K)

        correction = self.fno(R_in)

        # Residual connection
        R_current = R[:, -1].permute(0, 4, 1, 2, 3)
        H = R_current + correction

        return H.permute(0, 2, 3, 4, 1)


# =========================
# Linear Algebra
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
    NT, MT = F.shape
    NR, MR = W.shape

    if MR < NR:
        GL = W
    else:
        GL = torch.linalg.inv(W @ W.conj().T) @ W

    if MT < NT:
        GR = F.conj().T
    else:
        GR = F.conj().T @ torch.linalg.inv(F @ F.conj().T)

    return GL, GR


# =========================
# Channel Generator
# =========================

def steering_vector(N, angle):
    n = torch.arange(N)
    return torch.exp(1j * math.pi * n * torch.sin(angle))


def generate_channel(B, NR, NT, K, L=5):
    H = torch.zeros(B, NR, NT, K, dtype=torch.cfloat)

    for l in range(L):
        alpha = (torch.randn(B) + 1j * torch.randn(B)) * math.exp(-0.5*l)
        theta = torch.rand(B) * math.pi - math.pi / 2
        phi = torch.rand(B) * math.pi - math.pi / 2
        tau = torch.rand(B) * 5

        for b in range(B):
            a_tx = steering_vector(NT, theta[b])
            a_rx = steering_vector(NR, phi[b])

            for k in range(K):
                phase = torch.exp(-1j * 2 * math.pi * tau[b] * k / K)
                H[b, :, :, k] += alpha[b] * phase * torch.outer(a_rx, a_tx.conj())

    return torch.stack([H.real, H.imag], dim=-1)


# =========================
# Measurement
# =========================

def generate_measurement(H, F, W, noise_std):
    B, NR, NT, K, _ = H.shape
    Hc = H[..., 0] + 1j * H[..., 1]

    Y = []

    for k in range(K):
        Hk = Hc[:, :, :, k]

        Yk = torch.matmul(W.conj().T.unsqueeze(0), Hk)
        Yk = torch.matmul(Yk, F.unsqueeze(0))

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
        Rk = torch.matmul(GL.unsqueeze(0), Yk)
        Rk = torch.matmul(Rk, GR.unsqueeze(0))
        R.append(Rk)

    R = torch.stack(R, dim=-1)
    return torch.stack([R.real, R.imag], dim=-1)


# =========================
# SPR Data
# =========================

def generate_spr_data(B, D, NR, NT, K, rho, noise_std):
    H_t = generate_channel(B, NR, NT, K)

    R_list = []

    for t in range(D):
        innovation = generate_channel(B, NR, NT, K)
        H_t = rho * H_t + math.sqrt(1 - rho**2) * innovation

        if t == 0:
            MT, MR = NT, NR
        else:
            MT, MR = M_T, M_R

        F, W = generate_beams(NT, NR, MT, MR)
        GL, GR = compute_GL_GR(F, W)

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
# Training (PER-SNR)
# =========================

def train(model, device, steps, noise_std):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for step in range(steps):
        R, H = generate_spr_data(16, 4, N_R, N_T, 16, 0.95, noise_std)

        R, H = R.to(device), H.to(device)

        pred = model(R)
        loss = ((pred - H) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}, Loss {loss.item():.6f}")

    return model


# =========================
# Evaluation
# =========================

def evaluate(model, device, noise_std):
    model.eval()

    nmse_model = 0
    nmse_ls = 0

    with torch.no_grad():
        for _ in range(200):
            R, H = generate_spr_data(1, 4, 8, 8, 16, 0.95, noise_std)

            R, H = R.to(device), H.to(device)

            pred = model(R)
            H_ls = R[:, -1]

            nmse_model += nmse(pred, H).item()
            nmse_ls += nmse(H_ls, H).item()

    return nmse_model / 200, nmse_ls / 200


# =========================
# Main (PER-SNR LOOP)
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    snr_db_list = [0, 5, 10, 15, 20]
    noise_list = [10 ** (-snr / 20) for snr in snr_db_list]

    res_model = []
    res_ls = []

    for snr_db, noise_std in zip(snr_db_list, noise_list):
        print(f"\n=== SNR {snr_db} dB ===")

        model = SPR_FNO().to(device)

        # Train per SNR
        model = train(model, device, steps=200, noise_std=noise_std)

        nmse_m, nmse_l = evaluate(model, device, noise_std)

        res_model.append(to_db(nmse_m))
        res_ls.append(to_db(nmse_l))

        print(f"SPR-FNO NMSE: {nmse_m:.6f}")
        print(f"LS NMSE:      {nmse_l:.6f}")

    plt.plot(snr_db_list, res_model, 'o-', label="SPR-FNO")
    plt.plot(snr_db_list, res_ls, 'x--', label="LS")

    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("SPR-FNO vs LS (PER-SNR TRAINING)")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()