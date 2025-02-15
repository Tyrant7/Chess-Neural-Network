import torch
import random
import timeit
from torch.utils.data import (DataLoader, Dataset)
from torch import nn

piece_map = {c: i for i, c in enumerate("pnbrqk")}
piece_count = len(piece_map)


class ChessDataset(Dataset):
    def __init__(self, file_path, train=True, allow_data_augmentation=False):
        self.file_path = file_path
        self.allow_data_augmentation = allow_data_augmentation
        with open(file_path, "r") as file:
            fens = file.readlines()
            train_test_split = int(len(fens) * 0.8)
            self.fens = fens[:train_test_split] if train else fens[train_test_split:]

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]

        # TODO: More data argumentation
        # And flip boards vertically + piece colours + winner label
        # Diagonal flips when no pawns present for either side

        # TODO: Include separate plains for castling rights and 50 move counter and other info

        # We're going to encode the game from who ever is the side to move's perspective
        parts = fen.split(" ")
        board, side_to_move, castling_rights, score = parts[0], parts[1], parts[2], parts[-1]
        white_to_move = side_to_move == "w"
        horizontal_flip = (castling_rights == "-" and
                           self.allow_data_augmentation and
                           random.random() < 0.5)

        # Generate a 3D tensor for each board representation where a 1 represents a piece in that position
        # The 3D input data can take advantage of preserving pieces' spacial positions on the board through a CNN
        # (width, height, pieceType)
        data = torch.zeros((piece_count * 2, 8, 8), dtype=torch.float)
        x, y = 0, 0
        for c in board:
            if c.isalpha():
                if white_to_move:
                    colour_offset = 0 if c.isupper() else piece_count
                else:
                    colour_offset = piece_count if c.isupper() else 0
                y_offset = y if white_to_move else 7 - y
                x_offset = 7 - x if horizontal_flip else x
                data[piece_map[c.lower()] + colour_offset, y_offset, x_offset] = 1
                x += 1
            elif c.isdigit():
                x += int(c)
            elif c == "/":
                y += 1
                x = 0

        # Capture the game's score from the end of the fen
        # Remove brackets and newline
        label = score[1:-2]
        if white_to_move:
            label_tensor = torch.tensor(
                0 if label == "1.0" else 1 if label == "0.0" else 2
            )
        else:
            label_tensor = torch.tensor(
                1 if label == "1.0" else 0 if label == "0.0" else 2
            )
        return data, label_tensor


# TODO: Experiment with bottleneck residual blocks design (i.e. retaining input and output size, with a third,
# inner layer with a reduced number of features
# https://medium.com/@neetu.sigger/a-comprehensive-guide-to-understanding-and-implementing-bottleneck-residual-blocks-6b420706f66b
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += residual

        # Apply activation function at the end to prevent negatives
        # from leaking through after restoring residual
        return self.relu(x)


class ChessNetwork(nn.Module):
    def __init__(self, residual_layer_count=4):
        super().__init__()

        # (batch_size, 12, 8, 8) -> (batch_size, 128, 8, 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(piece_count * 2, 256, kernel_size=3, padding=1),
            # Batch norm won't change the size, but it can help stabilize learning
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Residual blocks will retain input dimensions
        # (batch_size, 128, 8, 8) -> (batch_size, 128, 8, 8)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(residual_layer_count)
        ])

        # (batch_size, 128, 8, 8) -> (batch_size, 128, 4, 4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # (batch_size, 512, 4, 4) -> (batch_size, 8192)
        # TODO: Experiment with global average pooling instead of only flattening data
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()

        # (batch_size, 8192) -> (batch_size, 512)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Output -> (win, loss, draw)
            nn.Linear(512, 3),
        )

    def forward(self, x):
        # Input convolution
        x = self.conv1(x)

        # All of our residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        # Down-sampling (no residual connected)
        x = self.conv2(x)

        # Flatten and fully-connected layer
        return self.fc(self.flat(x))


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
            print(f"Loss {loss:>7f} [{current:>7d}/{size:>7d}]")


def test_loop(dataloader, model, device, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            test_loss += loss_fn(y_pred, y).item()
            accuracy += (y_pred.argmax(1) == y).sum().item()

    test_loss /= num_batches
    accuracy /= size
    print(f"Test results: \nAccuracy: {(accuracy * 100):>0.1f}%, Avg Loss: {test_loss:>7f}")
    with open("accuracies.txt", "a") as f:
        f.write(str(accuracy))


if __name__ == '__main__':
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    BATCH_SIZE = 1024

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
    model_path = "networks/model_0.pt"

    data_path = "data/lichess-big3-resolved.book"
    train_dataset = ChessDataset(data_path, True, True)
    test_dataset = ChessDataset(data_path, False, True)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ChessNetwork(8).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    start_time = timeit.default_timer()
    print(f"Beginning training on device: {device}")
    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, device, loss_fn, optimizer)
        test_loop(test_dataloader, model, device, loss_fn)
        scheduler.step()

        print(f"Saving network state dict to {model_path}\n")
        torch.save(model.state_dict(), model_path)

    print(f"Completed training in {((timeit.default_timer() - start_time) * 1000):>.0f} ms")
