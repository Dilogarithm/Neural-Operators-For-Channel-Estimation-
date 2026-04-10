import torch
import torch.nn as nn
from neuralop.models import GNO


# =========================
# Model
# =========================

class SFT_GNO(nn.Module):
    def __init__(self, NR, NT, K, hidden_channels=32):
        super().__init__()

        self.NR = NR
        self.NT = NT
        self.K = K
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.gno = GNO(
            hidden_channels=hidden_channels,
            in_channels=4,
            out_channels=2,
        )

    def forward(self, R):
        """
        R: (B, T, NR, NT, K, 2)
        """
        # 🔹 Extract frames
        R_current = R[:, -1]   # (B, NR, NT, K, 2)
        R_prev    = R[:, -2]   # (B, NR, NT, K, 2)

        # 🔹 Convert to channels-first for FNO
        # (B, NR, NT, K, 2) → (B, 2, NR, NT, K)
        R_current_c = R_current.permute(0, 4, 1, 2, 3)
        R_prev_c    = R_prev.permute(0, 4, 1, 2, 3)

        # 🔹 Concatenate current + previous
        x = torch.cat([R_current_c, R_prev_c], dim=1)  # (B, 4, NR, NT, K)

        # 🔹 FNO correction
        correction = self.gno(x)  # (B, 2, NR, NT, K)

        # 🔹 Controlled residual
        H_c = R_current_c + self.alpha * correction

        # 🔹 Back to original format
        H = H_c.permute(0, 2, 3, 4, 1)  # (B, NR, NT, K, 2)

        return H


# =========================
# Synthetic Channel Generator
# =========================

def generate_channel(B, NR, NT, K, L=3):
    """
    Simple multi-path channel model
    """
    H = torch.zeros(B, NR, NT, K, 2)

    for _ in range(L):
        alpha = torch.randn(B, 1, 1, 1, 2)
        spatial = torch.randn(B, NR, NT, 1, 2)
        freq = torch.randn(B, 1, 1, K, 2)

        H += alpha * spatial * freq

    return H

def generate_data(B, T, NR, NT, K, noise_std=0.1, rho=0.90):
    """
    Generate slowly varying channel + noisy observations
    """

    # 🔹 Initial channel
    H_t = generate_channel(B, NR, NT, K)

    R_list = []
    H_list = []

    for t in range(T):
        # 🔹 evolve channel
        innovation = generate_channel(B, NR, NT, K)
        H_t = rho * H_t + (1 - rho) * innovation

        # 🔹 observation
        noise = noise_std * torch.randn_like(H_t)
        R_t = H_t + noise

        R_list.append(R_t)
        H_list.append(H_t)

    R = torch.stack(R_list, dim=1)   # (B, T, ...)
    H = H_list[-1]                   # target = last channel

    return R, H

# =========================
# Training
# =========================

def complex_mse(pred, target):
    return ((pred - target) ** 2).mean()


def main():
    # config
    B = 16
    T = 2
    NR = 8
    NT = 8
    K = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SFT_GNO(NR, NT, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...\n")

    for step in range(1000):
        R, H = generate_data(B, T, NR, NT, K)

        R = R.to(device)
        H = H.to(device)

        pred = model(R)
        loss = complex_mse(pred, H)

        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            H_avg = 0.5 * (R[:, -1] + R[:, -2])
            loss_avg = ((H_avg - H)**2).mean()
            print(f"loss_avg is{loss_avg}")
            print(f"Step {step}, Loss: {loss.item():.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

