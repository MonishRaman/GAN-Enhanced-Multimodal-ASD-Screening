import numpy as np
import torch
from torch.utils.data import Dataset


class GeneticDataset(Dataset):

    def __init__(self, X, y):

        self.X = np.asarray(X, dtype=np.float32)

        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        x = torch.tensor(
            self.X[index],
            dtype=torch.float32
        )

        y = torch.tensor(
            self.y[index],
            dtype=torch.long
        )

        return x, y