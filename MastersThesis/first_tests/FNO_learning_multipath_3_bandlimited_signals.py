
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

def main():

    device = "cpu"

    # %%
    # .. raw:: html
    #
    #    <div style="margin-top: 3em;"></div>
    #
    # Loading the dataset
    # ------------------------------

    
    N = 256
    x = torch.linspace(0, 1, N)
        
    # Fixed channel parameters
    K = 7
    alphas = torch.tensor([0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1])
    taus = torch.tensor([10, 30, 70, 100, 120, 150, 190])  # fixed delays

    def apply_operator(a):
        u = torch.zeros_like(a)

        for k in range(K):
            u += alphas[k] * torch.roll(a, shifts=int(taus[k]))

        return u


    from torch.utils.data import Dataset
    
    def smooth_random_signal(N, cutoff=5):
        A = torch.zeros(N, dtype=torch.complex64)
        A[:cutoff] = torch.randn(cutoff) + 1j * torch.randn(cutoff)
        a = torch.fft.ifft(A).real
        return a / a.abs().max()  # normalize
    
    def random_bandlimited_signal(N, B=20):
        x = torch.linspace(0,1,N)
        signal = torch.zeros(N)
        for k in range(-B, B):
            amp = torch.randn(1)
            signal += amp * torch.cos(2 * torch.pi * k * x)
        return signal


    class SimpleOperatorDataset(Dataset):
        def __init__(self, n_samples):
            self.inputs = []
            self.outputs = []

            for _ in range(n_samples):
                a = random_bandlimited_signal(N)
                u = apply_operator(a)
                self.inputs.append(a.unsqueeze(0))
                self.outputs.append(u.unsqueeze(0))

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return {
                "x": self.inputs[idx],
                "y": self.outputs[idx]
            }

    from torch.utils.data import DataLoader

    train_dataset = SimpleOperatorDataset(1000)
    test_dataset = SimpleOperatorDataset(200)

    train_loader = DataLoader(train_dataset, batch_size=32)
    test_loaders = {N: DataLoader(test_dataset, batch_size=32)}



    # %%
    # .. raw:: html
    #
    #    <div style="margin-top: 3em;"></div>
    #
    # Creating the FNO model
    # ----------------------

    model = FNO(
        n_modes=(24,),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
    )
    model = model.to(device)

    # Count and display the number of parameters
    n_params = count_model_params(model)
    print(f"\nOur model has {n_params} parameters.")
    sys.stdout.flush()


    # %%
    # .. raw:: html
    #
    #    <div style="margin-top: 3em;"></div>
    #
    # Creating the optimizer and scheduler
    # ------------------------------------
    # We use AdamW optimizer with weight decay for regularization
    optimizer = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # %%
    # .. raw:: html
    #
    #    <div style="margin-top: 3em;"></div>
    #
    # Setting up loss functions
    # -------------------------
    # We use H1 loss for training and L2 loss for evaluation
    # H1 loss is particularly good for PDE problems as it penalizes both function values and gradients


    l2loss = LpLoss(d=1, p=2)
    train_loss = l2loss
    eval_losses = {"l2": l2loss}


    # %%
    # .. raw:: html
    #
    #    <div style="margin-top: 3em;"></div>
    #
    # Training the model
    # ---------------------
    # We display the training configuration and then train the model

    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    sys.stdout.flush()

    # %%
    # .. raw:: html
    #
    #    <div style="margin-top: 3em;"></div>
    #
    # Creating the trainer
    # --------------------
    # We create a Trainer object that handles the training loop, evaluation, and logging
    
    trainer = Trainer(
        model=model,
        n_epochs=25,
        device=device,
        data_processor=None,
        wandb_log=False,  # Disable Weights & Biases logging for this tutorial
        eval_interval=5,  # Evaluate every 5 epochs
        use_distributed=False,  # Single GPU/CPU training
        verbose=True,  # Print training progress
    )

    # %%
    # .. raw:: html
    #
    #    <div style="margin-top: 3em;"></div>
    #
    # Training the model
    # ------------------
    # We train the model on our Darcy-Flow dataset. The trainer will:
    # 1. Run the forward pass through the FNO
    # 2. Compute the H1 loss
    # 3. Backpropagate and update weights
    # 4. Evaluate on test data every 3 epochs

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=train_loss,
        eval_losses=eval_losses,
    )

    # %%
    # .. raw:: html
    #
    #    <div style="margin-top: 3em;"></div>
    #
    # .. _plot_preds :
    # Visualizing predictions
    # ------------------------
    # Let's take a look at what our model's predicted outputs look like.
    # We wll compare the inputs, ground-truth outputs, and model predictions side by side.
    #
    # Note that in this example, we train on a very small resolution for
    # a very small number of epochs. In practice, we would train at a larger
    # resolution on many more samples.


    test_samples = test_loaders[N].dataset

    fig, axes = plt.subplots(3, 3, figsize=(10, 6))

    for index in range(3):
        data = test_samples[index]

        x = data["x"]               # shape (1, N)
        y = data["y"]               # shape (1, N)
        out = model(x.unsqueeze(0)) # shape (1, 1, N)

        x_np = x.squeeze().numpy()
        y_np = y.squeeze().numpy()
        out_np = out.detach().squeeze().numpy()

        # Input
        axes[index, 0].plot(x_np)
        if index == 0:
            axes[index, 0].set_title("Input")

        # Ground truth
        axes[index, 1].plot(y_np)
        if index == 0:
            axes[index, 1].set_title("Ground Truth")

        # Prediction
        axes[index, 2].plot(out_np)
        if index == 0:
            axes[index, 2].set_title("Prediction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
