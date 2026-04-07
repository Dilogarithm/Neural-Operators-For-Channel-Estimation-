import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from neuralop.models import FNO


# =========================
# Model
# =========================

class SFT_FNO(nn.Module):
    def __init__(self, NR, NT, K, hidden_channels=32):
        super().__init__()

        self.NR = NR
        self.NT = NT
        self.K = K

        self.fno = FNO(
            n_modes=(8, 8, 8),
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

        H = H_c.permute(0, 2, 3, 4, 1)

        return H


# =========================
# Physical Channel Generator
# =========================

def steering_vector(N, angle):
    n = torch.arange(N)
    return torch.exp(1j * math.pi * n * torch.sin(angle))


def generate_physical_channel(B, NR, NT, K, L=3):
    H = torch.zeros(B, NR, NT, K, dtype=torch.cfloat)

    for _ in range(L):
        alpha = torch.randn(B) + 1j * torch.randn(B)

        theta = torch.rand(B) * math.pi - math.pi/2
        phi = torch.rand(B) * math.pi - math.pi/2
        tau = torch.rand(B) * 1e-6

        for b in range(B):
            a_tx = steering_vector(NT, theta[b])
            a_rx = steering_vector(NR, phi[b])

            for k in range(K):
                phase = torch.exp(-1j * 2 * math.pi * k * tau[b])
                H[b, :, :, k] += alpha[b] * torch.outer(a_rx, a_tx.conj()) * phase

    H_real = torch.stack([H.real, H.imag], dim=-1)
    return H_real


def generate_realistic_data(B, T, NR, NT, K, rho=0.95, noise_std=0.1):
    H_t = generate_physical_channel(B, NR, NT, K)

    R_list = []
    H_list = []

    for _ in range(T):
        innovation = generate_physical_channel(B, NR, NT, K)
        H_t = rho * H_t + (1 - rho) * innovation

        noise = noise_std * torch.randn_like(H_t)
        R_t = H_t + noise

        R_list.append(R_t)
        H_list.append(H_t)

    R = torch.stack(R_list, dim=1)
    H = H_list[-1]

    return R, H


# =========================
# Loss
# =========================

def complex_mse(pred, target):
    return ((pred - target) ** 2).mean()


# =========================
# Visualization
# =========================

def plot_channel(H_true, H_pred):
    H_true_c = H_true[..., 0] + 1j * H_true[..., 1]
    H_pred_c = H_pred[..., 0] + 1j * H_pred[..., 1]

    H_true_mag = torch.abs(H_true_c).cpu().numpy()
    H_pred_mag = torch.abs(H_pred_c).detach().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.imshow(H_true_mag[0], aspect='auto')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.imshow(H_pred_mag[0], aspect='auto')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Error")
    plt.imshow(abs(H_true_mag[0] - H_pred_mag[0]), aspect='auto')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# =========================
# Training
# =========================

def main():
    B = 16
    T = 2
    NR = 8
    NT = 8
    K = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SFT_FNO(NR, NT, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training...\n")

    for step in range(1000):
        R, H = generate_realistic_data(B, T, NR, NT, K, rho=0.9)

        R = R.to(device)
        H = H.to(device)

        pred = model(R)
        loss = complex_mse(pred, H)

        # baseline
        H_avg = 0.5 * (R[:, -1] + R[:, -2])
        loss_avg = ((H_avg - H) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}")
            print(f"Model Loss: {loss.item():.6f}")
            print(f"Avg Loss:   {loss_avg.item():.6f}")
            print(f"alpha:      {model.alpha.item():.4f}")
            print("------")

    print("\nTraining done.\n")

    # =========================
    # Visualization
    # =========================

    R, H = generate_realistic_data(1, T, NR, NT, K, rho=0.9)

    R = R.to(device)
    H = H.to(device)

    pred = model(R)

    plot_channel(H[0], pred[0])


if __name__ == "__main__":
    main()
