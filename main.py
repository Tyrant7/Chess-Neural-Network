import torch
import re
from torch.utils.data import (DataLoader, Dataset)
from torch import nn

piece_map = {c: i for i, c in enumerate("pnbrqk")}
piece_count = len(piece_map)

class ChessDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, "r") as file:
            self.fens = file.readlines()

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]

        # Generate a 3D tensor for each board representation where a 1 represents a piece in that position
        # The 3D input data can take advantage of preserving pieces' spacial positions on the board through a CNN
        # (width, height, pieceType)
        data = torch.zeros((piece_count * 2, 8, 8), dtype=torch.float)
        x, y = 0, 0
        for c in fen:
            # A space marks the end of the board representation of the fen
            if c == " ":
                break
            if c.isalpha():
                colour_offset = 0 if c.isupper() else piece_count
                piece_idx = piece_map[c.lower()]
                data[piece_idx + colour_offset, y, x] = 1
                x += 1
            elif c.isdigit():
                x += int(c)
            elif c == "/":
                y += 1
                x = 0

        # Capture the game's score from the end of the fen
        label = re.search(r"\[(\d\.\d)]", fen).group(1)

        return data, float(label)


class ChessNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # (batch_size, 12, 8, 8) -> (batch_size, 32, 6, 6)
            nn.Conv2d(piece_count * 2, 32, kernel_size=3),
            nn.ReLU(),
            # (batch_size, 32, 6, 6) -> (batch_size, 128, 4, 4)
            nn.Conv2d(32, 128, kernel_size=3),
            nn.ReLU(),
            # (batch_size, 128, 4, 4) -> (batch_size, 2048)
            nn.Flatten(),
            nn.Linear(128*4*4, 1),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    data_path = "data/lichess-big3-resolved.book"
    dataset = ChessDataset(data_path)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    data, label = next(iter(dataloader))
    print(data, label)

    model = ChessNetwork()
    output = model(data)
    print(output)
