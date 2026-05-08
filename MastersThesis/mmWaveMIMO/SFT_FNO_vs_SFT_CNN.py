import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from neuralop.models import FNO


# =========================================================
# TRUE PAPER-STYLE SFT-CNN
# =========================================================
#
# IMPORTANT:
# The paper uses 2D convolutions.
#
# Frequency + temporal information are encoded
# as CHANNELS, not spatial dimensions.
#
# Input channels:
#   2 temporal states
# x 2 complex parts
# x K subcarriers
#
# => 4K channels
#
# Spatial dimensions:
#   NR x NT
#
# =========================================================

class SFT_CNN(nn.Module):

    def __init__(self, K):

        super().__init__()

        self.K = K

        in_channels = 4 * K

        out_channels = 2 * K

        self.conv_in = nn.Sequential(

            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(64)
        )

        self.hidden = nn.ModuleList([

            nn.Sequential(

                nn.Conv2d(
                    64,
                    64,
                    kernel_size=3,
                    padding=1
                ),

                nn.ReLU(),

                nn.BatchNorm2d(64)

            )

            for _ in range(8)

        ])

        self.conv_out = nn.Conv2d(
            64,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, R):

        # ============================================
        # INPUT:
        # (B, 2, NR, NT, K, 2)
        # ============================================

        B, _, NR, NT, K, _ = R.shape

        R_prev = R[:, 0]
        R_curr = R[:, 1]

        # --------------------------------------------
        # Convert:
        #
        # (B, NR, NT, K, 2)
        #
        # -> (B, 2K, NR, NT)
        #
        # real/imag become channels
        # --------------------------------------------

        R_prev = R_prev.permute(
            0, 3, 4, 1, 2
        ).reshape(
            B,
            2 * K,
            NR,
            NT
        )

        R_curr = R_curr.permute(
            0, 3, 4, 1, 2
        ).reshape(
            B,
            2 * K,
            NR,
            NT
        )

        # --------------------------------------------
        # concatenate temporal states
        #
        # => (B, 4K, NR, NT)
        # --------------------------------------------

        x = torch.cat(
            [R_prev, R_curr],
            dim=1
        )

        x = self.conv_in(x)

        for layer in self.hidden:
            x = layer(x)

        correction = self.conv_out(x)

        # --------------------------------------------
        # residual refinement
        # --------------------------------------------

        H_est = R_curr + correction

        # --------------------------------------------
        # Convert back:
        #
        # (B, 2K, NR, NT)
        #
        # -> (B, NR, NT, K, 2)
        # --------------------------------------------

        H_est = H_est.reshape(
            B,
            K,
            2,
            NR,
            NT
        )

        H_est = H_est.permute(
            0,
            3,
            4,
            1,
            2
        )

        return H_est


# =========================================================
# SFT-FNO
# =========================================================

class SFT_FNO(nn.Module):

    def __init__(self, hidden_channels=64):

        super().__init__()

        self.fno = FNO(

            n_modes=(8, 8, 8),

            hidden_channels=hidden_channels,

            in_channels=4,

            out_channels=2,
        )

        self.beta = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(self, R):

        # (B, 2, NR, NT, K, 2)

        R_prev = R[:, 0]
        R_curr = R[:, 1]

        # -> (B, 2, NR, NT, K)

        R_prev = R_prev.permute(
            0, 4, 1, 2, 3
        )

        R_curr = R_curr.permute(
            0, 4, 1, 2, 3
        )

        # -> (B, 4, NR, NT, K)

        x = torch.cat(
            [R_prev, R_curr],
            dim=1
        )

        correction = self.fno(x)

        H_est = (
            R_curr
            + self.beta * correction
        )

        return H_est.permute(
            0,
            2,
            3,
            4,
            1
        )


# =========================================================
# DFT MATRICES
# =========================================================

def dft_matrix(N):

    n = torch.arange(N)

    k = n.view(-1, 1)

    W = torch.exp(
        -2j * math.pi * k * n / N
    )

    return W / math.sqrt(N)


def generate_beams(NT, NR, MT, MR):

    F = dft_matrix(NT)[:, :MT]

    W = dft_matrix(NR)[:, :MR]

    return F, W


def compute_GL_GR(F, W):

    GL = (
        torch.linalg.inv(
            W @ W.conj().T
        ) @ W
    )

    GR = (
        F.conj().T
        @ torch.linalg.inv(
            F @ F.conj().T
        )
    )

    return GL, GR


# =========================================================
# MEASUREMENTS
# =========================================================

def generate_measurement(
    H,
    F,
    W,
    noise_std
):

    B, NR, NT, K, _ = H.shape

    Hc = (
        H[..., 0]
        + 1j * H[..., 1]
    )

    Y = []

    for k in range(K):

        Hk = Hc[:, :, :, k]

        Yk = (
            W.conj().T
            @ Hk
            @ F
        )

        noise = noise_std * (
            torch.randn_like(Yk)
            + 1j * torch.randn_like(Yk)
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

    return torch.stack(
        [R.real, R.imag],
        dim=-1
    )


# =========================================================
# PHYSICAL CHANNEL
# =========================================================

def steering_vector(N, angle):

    n = torch.arange(N)

    return torch.exp(
        1j
        * math.pi
        * n
        * torch.sin(angle)
    )


def generate_physical_channel(
    B,
    NR,
    NT,
    K,
    L=3
):

    H = torch.zeros(
        B,
        NR,
        NT,
        K,
        dtype=torch.cfloat
    )

    fc = 28e9

    delta_f = 15e3

    for l in range(L):

        alpha = (
            torch.randn(B)
            + 1j * torch.randn(B)
        ) * math.exp(-0.5 * l)

        theta = (
            torch.rand(B)
            * math.pi
            - math.pi / 2
        )

        phi = (
            torch.rand(B)
            * math.pi
            - math.pi / 2
        )

        tau = (
            torch.rand(B)
            * 200e-9
        )

        for b in range(B):

            a_tx = steering_vector(
                NT,
                theta[b]
            )

            a_rx = steering_vector(
                NR,
                phi[b]
            )

            for k in range(K):

                f_k = (
                    fc
                    + k * delta_f
                )

                phase = torch.exp(
                    -1j
                    * 2
                    * math.pi
                    * f_k
                    * tau[b]
                )

                H[b, :, :, k] += (

                    alpha[b]

                    * torch.outer(
                        a_rx,
                        a_tx.conj()
                    )

                    * phase
                )

    return torch.stack(
        [H.real, H.imag],
        dim=-1
    )


# =========================================================
# TEMPORAL EVOLUTION
# =========================================================

def generate_sample(
    B,
    NR,
    NT,
    K,
    rho,
    noise_std,
    F,
    W,
    GL,
    GR
):

    H_prev = generate_physical_channel(
        B,
        NR,
        NT,
        K
    )

    innovation = generate_physical_channel(
        B,
        NR,
        NT,
        K
    )

    H_curr = (

        rho * H_prev

        + math.sqrt(
            1 - rho**2
        ) * innovation

    )

    Y_prev = generate_measurement(
        H_prev,
        F,
        W,
        noise_std
    )

    Y_curr = generate_measurement(
        H_curr,
        F,
        W,
        noise_std
    )

    R_prev = compute_R(
        Y_prev,
        GL,
        GR
    )

    R_curr = compute_R(
        Y_curr,
        GL,
        GR
    )

    # (B, 2, NR, NT, K, 2)

    R = torch.stack(
        [R_prev, R_curr],
        dim=1
    )

    return R, H_curr


# =========================================================
# TRAIN
# =========================================================

def train_model(
    model,
    noise_std,
    NR,
    NT,
    K,
    rho,
    device,
    F,
    W,
    GL,
    GR,
    steps=500
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4
    )

    model.train()

    for step in range(steps):

        R, H = generate_sample(
            16,
            NR,
            NT,
            K,
            rho,
            noise_std,
            F,
            W,
            GL,
            GR
        )

        R = R.to(device)

        H = H.to(device)

        scale = (
            R.abs().mean()
            + 1e-6
        )

        R = R / scale

        H = H / scale

        pred = model(R)

        loss = (
            (pred - H) ** 2
        ).mean()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if step % 100 == 0:

            print(
                f"Step {step}, "
                f"Loss {loss.item():.6f}"
            )

    return model


# =========================================================
# NMSE
# =========================================================

def nmse(pred, target):

    return (

        ((pred - target) ** 2).sum()

        / (
            (target ** 2).sum()
            + 1e-12
        )

    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(
    model,
    noise_std,
    NR,
    NT,
    K,
    rho,
    device,
    F,
    W,
    GL,
    GR
):

    model.eval()

    nmse_model = 0

    nmse_te = 0

    with torch.no_grad():

        for _ in range(300):

            R, H = generate_sample(
                1,
                NR,
                NT,
                K,
                rho,
                noise_std,
                F,
                W,
                GL,
                GR
            )

            R = R.to(device)

            H = H.to(device)

            scale = (
                R.abs().mean()
                + 1e-6
            )

            R = R / scale

            H = H / scale

            pred = model(R)

            H_te = R[:, 1]

            nmse_model += nmse(
                pred,
                H
            ).item()

            nmse_te += nmse(
                H_te,
                H
            ).item()

    return (
        nmse_model / 300,
        nmse_te / 300
    )


# =========================================================
# MAIN
# =========================================================

def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    NR = 6
    NT = 6
    K = 16

    rho = 0.9

    F, W = generate_beams(
        NT,
        NR,
        NT,
        NR
    )

    GL, GR = compute_GL_GR(F, W)

    F = F.to(device)

    W = W.to(device)

    GL = GL.to(device)

    GR = GR.to(device)

    snr_db_list = [
        0,
        5,
        10,
        15,
        20
    ]

    noise_list = [

        10 ** (-snr / 20)

        for snr in snr_db_list

    ]

    results_fno = []

    results_cnn = []

    results_te = []

    for snr_db, noise_std in zip(
        snr_db_list,
        noise_list
    ):

        print("\n")
        print("=" * 50)
        print(f"SNR = {snr_db} dB")
        print("=" * 50)

        # =============================================
        # FNO
        # =============================================

        print("\nTraining SFT-FNO...\n")

        model_fno = SFT_FNO().to(device)

        model_fno = train_model(

            model_fno,

            noise_std,

            NR,
            NT,
            K,

            rho,

            device,

            F,
            W,
            GL,
            GR

        )

        # =============================================
        # CNN
        # =============================================

        print("\nTraining SFT-CNN...\n")

        model_cnn = SFT_CNN(K).to(device)

        model_cnn = train_model(

            model_cnn,

            noise_std,

            NR,
            NT,
            K,

            rho,

            device,

            F,
            W,
            GL,
            GR

        )

        # =============================================
        # EVALUATION
        # =============================================

        nmse_fno, nmse_te = evaluate_model(

            model_fno,

            noise_std,

            NR,
            NT,
            K,

            rho,

            device,

            F,
            W,
            GL,
            GR

        )

        nmse_cnn, _ = evaluate_model(

            model_cnn,

            noise_std,

            NR,
            NT,
            K,

            rho,

            device,

            F,
            W,
            GL,
            GR

        )

        print("\nResults:")
        print(f"SFT-FNO NMSE : {nmse_fno:.6f}")
        print(f"SFT-CNN NMSE : {nmse_cnn:.6f}")
        print(f"TE Baseline  : {nmse_te:.6f}")

        results_fno.append(
            10 * math.log10(nmse_fno)
        )

        results_cnn.append(
            10 * math.log10(nmse_cnn)
        )

        results_te.append(
            10 * math.log10(nmse_te)
        )

    # =================================================
    # PLOT
    # =================================================

    plt.figure(figsize=(8, 5))

    plt.plot(

        snr_db_list,

        results_fno,

        'o-',

        linewidth=2,

        label="SFT-FNO"

    )

    plt.plot(

        snr_db_list,

        results_cnn,

        's-',

        linewidth=2,

        label="Paper SFT-CNN"

    )

    plt.plot(

        snr_db_list,

        results_te,

        'x--',

        linewidth=2,

        label="Tentative Estimate"

    )

    plt.xlabel("SNR (dB)")

    plt.ylabel("NMSE (dB)")

    plt.title(
        "SFT-FNO vs Paper SFT-CNN"
    )

    plt.grid(True)

    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()