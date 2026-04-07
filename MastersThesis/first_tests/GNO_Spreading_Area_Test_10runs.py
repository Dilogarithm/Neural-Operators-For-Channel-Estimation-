import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

from neuralop.models import GNO
from neuralop.training import Trainer
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop.losses import LpLoss

from torch.utils.data import Dataset, DataLoader


# =========================
# Channel generator
# =========================
def generate_channel(N, max_delay, max_doppler, K=3):
    delays = torch.randint(0, max_delay, (K,))
    dopplers = torch.randint(-max_doppler, max_doppler, (K,))
    coeffs = torch.randn(K)

    def H(x):
        y = torch.zeros_like(x)

        for k in range(K):
            d = delays[k]
            nu = dopplers[k]

            shifted = torch.roll(x, shifts=int(d))
            phase = torch.exp(2j * torch.pi * nu * torch.arange(N) / N)

            y += coeffs[k] * shifted * phase.real

        return y

    return H


# =========================
# Dataset
# =========================
def smooth_random_signal(N, cutoff=5):
    A = torch.zeros(N, dtype=torch.complex64)
    A[:cutoff] = torch.randn(cutoff) + 1j * torch.randn(cutoff)
    a = torch.fft.ifft(A).real
    return a / (a.abs().max() + 1e-6)


class SimpleOperatorDataset(Dataset):
    def __init__(self, n_samples, N, operator):
        self.inputs = []
        self.outputs = []

        for _ in range(n_samples):
            a = smooth_random_signal(N)
            u = operator(a)

            self.inputs.append(a.unsqueeze(0))
            self.outputs.append(u.unsqueeze(0))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"x": self.inputs[idx], "y": self.outputs[idx]}


# =========================
# Experiment loop
# =========================
def run_experiment(max_delay, max_doppler):

    N = 256
    device = "cpu"

    H = generate_channel(N, max_delay, max_doppler)

    train_dataset = SimpleOperatorDataset(800, N, H)
    test_dataset = SimpleOperatorDataset(200, N, H)

    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # model
    model = GNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=20,
    ).to(device)

    # IMPORTANT: initialize dynamic weights
    model(torch.randn(1, 1, N))

    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    loss_fn = LpLoss(d=1, p=2)

    trainer = Trainer(
        model=model,
        n_epochs=20,
        device=device,
        wandb_log=False,
        eval_interval=5,
        verbose=False,
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders={N: test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=loss_fn,
        eval_losses={"l2": loss_fn},
    )

    # compute test error manually
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"]
            y = batch["y"]
            out = model(x)
            total_loss += ((out - y) ** 2).mean().item()

    return total_loss / len(test_loader)


# =========================
# MAIN
# =========================

def main():

    spreads = [
        (1, 1),
        (3, 3),
        (5, 5),
        (7, 7),
        (10, 10),
        (15, 15),
        (17,17),
        (20, 20),
        (23, 23),
        (24, 24),
        (25, 25),
        (26, 26),
        (27, 27),
        (28, 28),
        (29, 29),
        (30, 30),
    ]

    n_runs = 20

    mean_errors = []
    std_errors = []
    S_vals = []

    for D, F in spreads:
        print(f"\n=== Running D={D}, F={F} ===")

        run_errors = []

        for i in range(n_runs):
            print(f"  Run {i+1}/{n_runs}")

            err = run_experiment(D, F)
            run_errors.append(err)

        run_errors = np.array(run_errors)

        mean_err = run_errors.mean()
        std_err = run_errors.std()

        S = D * F
        S_vals.append(S)
        mean_errors.append(mean_err)
        std_errors.append(std_err)

        print(f"S={S}, mean error={mean_err:.6f}, std={std_err:.6f}")

    # plot (same style + error bars)
    plt.figure()
    plt.errorbar(S_vals, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
    plt.xlabel("Spreading area S = D * F")
    plt.ylabel("Test MSE")
    plt.title("GNO performance vs spreading (avg over 10 runs)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
