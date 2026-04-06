import torch
import torch.nn as nn


class GaborConv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_fft=64,
        hop_length=32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.window = torch.hann_window(n_fft)

        # will initialize after seeing STFT size
        self.weight = None

    def forward(self, x, **kwargs):
        # x: (B, C, T)
        B, C, T = x.shape
        device = x.device
        window = self.window.to(device)

        # --- STFT for all channels ---
        U = torch.stack([
            torch.stft(
                x[:, i, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
                center=True,
                normalized=True,
            )
            for i in range(self.in_channels)
        ], dim=1)
        # U: (B, in_channels, F, TT)

        B, Cin, F, TT = U.shape

        # --- initialize weight if needed ---
        if self.weight is None:
            self.weight = nn.Parameter(
                0.01*torch.randn(
                    self.out_channels,
                    self.in_channels,
                    F,
                    TT,
                    device=device
                )
            )

        # --- Gabor multiplier ---
        # (B, 1, in, F, T) * (1, out, in, F, T)
        Y = (U.unsqueeze(1) * self.weight.unsqueeze(0)).sum(dim=2)
        # Y: (B, out_channels, F, TT)

        # --- inverse STFT ---
        outputs = torch.stack([
            torch.istft(
                Y[:, o],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                length=T,
                normalized=True,
                center=True,
            )
            for o in range(self.out_channels)
        ], dim=1)

        # outputs: (B, out_channels, T)
        return outputs


class GNO(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=16,
        n_layers=2,
        n_fft=64,
        hop_length=32,
    ):
        super().__init__()

        self.lifting = nn.Conv1d(in_channels, hidden_channels, 1)

        self.layers = nn.ModuleList([
            GaborConv1D(
                hidden_channels,
                hidden_channels,
                n_fft=n_fft,
                hop_length=hop_length
            )
            for _ in range(n_layers)
        ])

        self.projection = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, **kwargs):
        # x: (B, in_channels, T)
        x = self.lifting(x)

        for layer in self.layers:
            x = x + layer(x)  # residual

        x = self.projection(x)
        return x
