import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMixer(nn.Module):
    """
    Mixes information across time dimension
    Input:  (B, T, D)
    Output: (B, T, D)
    """
    def __init__(self, T, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=T,
            out_channels=T,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        return self.conv(x)


class SpatialOperator(nn.Module):
    """
    Global operator (can later be replaced by FNO)
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)  # residual


class ContinuousSFTNO(nn.Module):
    """
    Continuous SFT Neural Operator
    Equivalent to SFT-CNN but in operator form

    Input:
        R: (B, T, NR, NT, K, 2)

    Output:
        H: (B, NR, NT, K, 2)
    """

    def __init__(
        self,
        NR,
        NT,
        K,
        T=2,
        hidden_dim=256,
    ):
        super().__init__()

        self.NR = NR
        self.NT = NT
        self.K = K
        self.T = T

        self.input_dim = 2 * NR * NT * K * 2
        self.output_dim = NR * NT * K * 2

        # 🔹 Lift (embedding)
        self.lift = nn.Linear(self.input_dim, hidden_dim)

        # 🔹 Temporal operator
        self.temporal = TemporalMixer(T)

        # 🔹 Spatial operator
        self.operator = SpatialOperator(hidden_dim, hidden_dim)

        # 🔹 Projection
        self.proj = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, R):
        """
        R: (B, T, NR, NT, K, 2)
        """
        B = R.shape[0]

        # 🔹 Extract current and previous frames
        R_current = R[:, -1]   # (B, NR, NT, K, 2)
        R_prev    = R[:, -2]   # (B, NR, NT, K, 2)

        # 🔹 Flatten
        R_current_flat = R_current.reshape(B, -1)
        R_prev_flat    = R_prev.reshape(B, -1)

        # 🔹 Combine both (like SFT-CNN input)
        x_input = torch.cat([R_current_flat, R_prev_flat], dim=-1)

        # 🔹 Lift
        x = self.lift(x_input)

        # 🔹 Operator (residual block)
        x = self.operator(x)

        # 🔹 Project back
        out = self.proj(x)

        # 🔥 Residual connection (anchor on current frame)
        out = out + R_current_flat

        # 🔹 Reshape back to channel format
        H = out.view(B, self.NR, self.NT, self.K, 2)

        return H