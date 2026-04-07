import torch
import torch.nn as nn
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

    def forward(self, R):
        """
        R: (B, T, NR, NT, K, 2)
        """
        # current + previous
        R_current = R[:, -1]
        R_prev = R[:, -2]

        # (B, NR, NT, K, 2) → (B, 2, NR, NT, K)
        R_current = R_current.permute(0, 4, 1, 2, 3)
        R_prev = R_prev.permute(0, 4, 1, 2, 3)

        # concatenate channels
        x = torch.cat([R_current, R_prev], dim=1)

        # FNO correction
        correction = self.fno(x)

        # residual
        H = correction + R_current

        # back to original format
        H = H.permute(0, 2, 3, 4, 1)

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


def generate_data(B, T, NR, NT, K, noise_std=0.1):
    """
    Generate noisy observations R from clean channel H
    """
    H = generate_channel(B, NR, NT, K)

    # repeat over time
    R = H.unsqueeze(1).repeat(1, T, 1, 1, 1, 1)

    # add noise
    R = R + noise_std * torch.randn_like(R)

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

    model = SFT_FNO(NR, NT, K).to(device)
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
            print(f"Step {step}, Loss: {loss.item():.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
