"""
I had ChatGPT make this. I wanted to know how to make a torch dataset and iterate over it.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # could be numpy array or torch tensor
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Example data
X = np.random.randn(100, 10).astype(np.float32)
y = np.random.randint(0, 2, size=100).astype(np.int64)

# Optional: convert to tensors up front
X = torch.from_numpy(X)
y = torch.from_numpy(y)

dataset = MyDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for xb, yb in loader:
    # use xb and yb in training loop
    print("x:", xb.shape, "y:", y.shape)

# x: torch.Size([16, 10]) y: torch.Size([100])
# x: torch.Size([16, 10]) y: torch.Size([100])
# x: torch.Size([16, 10]) y: torch.Size([100])
# x: torch.Size([16, 10]) y: torch.Size([100])
# x: torch.Size([16, 10]) y: torch.Size([100])
# x: torch.Size([16, 10]) y: torch.Size([100])
# x: torch.Size([4, 10]) y: torch.Size([100])
