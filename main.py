import torch
from torch.utils.data import (DataLoader, Dataset)

class ChessDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    print(torch.__version__)

