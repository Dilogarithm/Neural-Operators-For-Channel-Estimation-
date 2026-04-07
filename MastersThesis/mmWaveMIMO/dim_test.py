import torch
from neuralop.models import sft

# small test config
B = 2
T = 2
NR = 4
NT = 4
K = 8

model = sft.ContinuousSFTNO(NR, NT, K, T)

R = torch.randn(B, T, NR, NT, K, 2)

H = model(R)

print("Input shape:", R.shape)
print("Output shape:", H.shape)
