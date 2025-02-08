import torch
import re
import timeit
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
        label_tensor = torch.zeros(2, dtype=torch.float)
        if label == "1.0":
            label_tensor[0] = 1
        elif label == "0.0":
            label_tensor[1] = 1
        else:
            label_tensor[0] = 0.5
            label_tensor[1] = 0.5

        return data, label_tensor


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
            nn.Linear(128*4*4, 2),
        )

    def forward(self, x):
        return self.model(x)


def train_loop(dataloader, model, device, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Printout
        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(x)
            print(f"Loss {loss:>7f} [{current:>6d}/{size:>6d}]")


def test_loop(dataloader, model, device, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            test_loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAccuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>7f}\n")


if __name__ == '__main__':
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    BATCH_SIZE = 32

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

    data_path = "data/lichess-big3-resolved.book"
    dataset = ChessDataset(data_path)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ChessNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    start_time = timeit.default_timer()
    print(f"Beginning training on device: {device}")
    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n---------------------------")
        train_loop(dataloader, model, device, loss_fn, optimizer)
        test_loop(dataloader, model, device, loss_fn)
    print(f"Completed training in {((timeit.default_timer() - start_time) * 1000):>.0f} ms")
