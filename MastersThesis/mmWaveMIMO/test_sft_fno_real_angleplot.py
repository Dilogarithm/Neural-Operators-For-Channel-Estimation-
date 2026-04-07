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
# Physical Channel Generator (FIXED)
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

    H_real = torch.stack([H.real, H.imag], dim=-1)
    return H_real


def generate_realistic_data(B, T, NR, NT, K, rho=0.9, noise_std=0.1):
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
# Angle Dictionary Projection
# =========================

def build_steering_dictionary(N, num_angles=64):
    """
    Correct dictionary using sin(theta) grid
    """
    u = torch.linspace(-1, 1, num_angles)  # sin(theta)
    n = torch.arange(N)

    A = torch.exp(1j * math.pi * n[:, None] * u[None, :])  # (N, num_angles)

    return A


def project_to_angle_domain(H, A_rx, A_tx):
    A_rx_H = torch.conj(A_rx).T
    A_tx_H = torch.conj(A_tx).T

    H_angle = []

    for k in range(H.shape[-1]):
        Hk = H[:, :, k]
        H_proj = A_rx_H @ Hk @ A_tx
        H_angle.append(H_proj)

    H_angle = torch.stack(H_angle, dim=-1)
    return H_angle


# =========================
# Visualization (CORRECT)
# =========================

def plot_angle_domain(H_true, H_pred, NR, NT):
    H_true_c = H_true[..., 0] + 1j * H_true[..., 1]
    H_pred_c = H_pred[..., 0] + 1j * H_pred[..., 1]

    A_rx = build_steering_dictionary(NR).to(H_true.device)
    A_tx = build_steering_dictionary(NT).to(H_true.device)

    H_true_angle = project_to_angle_domain(H_true_c, A_rx, A_tx)
    H_pred_angle = project_to_angle_domain(H_pred_c, A_rx, A_tx)

    # ✅ TAKE SINGLE FREQUENCY (CRITICAL FIX)
    k = H_true_angle.shape[-1] // 2

    H_true_angle = torch.abs(H_true_angle[:, :, k])
    H_pred_angle = torch.abs(H_pred_angle[:, :, k])

    # log scale
    H_true_log = 20 * torch.log10(H_true_angle + 1e-6)
    H_pred_log = 20 * torch.log10(H_pred_angle + 1e-6)

    H_true_np = H_true_log.detach().cpu().numpy()
    H_pred_np = H_pred_log.detach().cpu().numpy()

    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.title("Ground Truth (Angle Domain)")
    plt.imshow(H_true_np, aspect='auto')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Prediction (Angle Domain)")
    plt.imshow(H_pred_np, aspect='auto')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

# =========================
# Training
# =========================

def main():
    B = 16
    T = 2
    NR = 32
    NT = 32
    K = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SFT_FNO(NR, NT, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training...\n")

    for step in range(600):
        R, H = generate_realistic_data(B, T, NR, NT, K)

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

    model.eval()

    with torch.no_grad():
        R, H = generate_realistic_data(1, T, NR, NT, K)

        R = R.to(device)
        H = H.to(device)

        pred = model(R)

        plot_angle_domain(H[0], pred[0], NR, NT)


if __name__ == "__main__":
    main()