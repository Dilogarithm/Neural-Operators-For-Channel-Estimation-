import torch
import torch.nn as nn
from neuralop.models import sft

B = 16
T = 2
NR = 4
NT = 4
K = 8

model = sft.ContinuousSFTNO(NR, NT, K, T)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    R = torch.randn(B, T, NR, NT, K, 2)
    
    # target = last frame
    H_target = R[:, -1]

    pred = model(R)

    loss = ((pred - H_target)**2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")
