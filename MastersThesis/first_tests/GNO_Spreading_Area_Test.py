import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

from gno import GNO
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

    N = 64
    device = "cpu"

    H = generate_channel(N, max_delay, max_doppler)

    train_dataset = SimpleOperatorDataset(1000, N, H)
    test_dataset = SimpleOperatorDataset(500, N, H)

    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # model
    model = GNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
    ).to(device)

    # IMPORTANT: initialize dynamic weights
    model(torch.randn(1, 1, N))

    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    loss_fn = LpLoss(d=1, p=2)

    trainer = Trainer(
        model=model,
        n_epochs=30,
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
        (5, 5),
        (10, 10),
        (15, 15),
        (20, 20),
        (25, 25),
        (30, 30),
    ]

    errors = []
    S_vals = []

    for D, F in spreads:
        print(f"Running D={D}, F={F}")

        err = run_experiment(D, F)

        S = D * F
        S_vals.append(S)
        errors.append(err)

        print(f"S={S}, error={err}")

    # plot
    plt.plot(S_vals, errors, 'o-')
    plt.xlabel("Spreading area S = D * F")
    plt.ylabel("Test MSE")
    plt.title("GNO performance vs spreading")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
