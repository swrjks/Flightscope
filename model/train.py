import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os

# ---------------- CNN Model ----------------
class CNN1DFlightRegime(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN1DFlightRegime, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # -> (batch, input_dim, seq_len)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool1d(x, 1)  # -> (batch, channels, 1)

        x = x.squeeze(-1)  # -> (batch, channels)
        x = self.dropout(F.relu(self.fc1(x)))
        out = self.fc2(x)  # -> (batch, num_classes)

        return out

# ---------------- Training Script ----------------
def train_model(train_data, test_data, num_classes, epochs=15, batch_size=64, lr=1e-3, device="cpu"):
    X_train, y_train = train_data
    X_test, y_test = test_data

    # convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    # dataloaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # model
    input_dim = X_train.shape[2]
    model = CNN1DFlightRegime(input_dim=input_dim, num_classes=num_classes).to(device)
    print(model)

    # loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs+1):
        # ---- Train ----
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total

        # ---- Eval ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                _, preds = torch.max(outputs, 1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch:02d}: Train Acc {train_acc:.3f}, Val Acc {val_acc:.3f}")

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model.pth")

    print(f"Training done. Best Val Acc: {best_acc:.3f}")
    return model

# ---------------- Main ----------------
if __name__ == "__main__":
    # load processed data
    train_npz = np.load("./dataset/processed/train.npz")
    test_npz  = np.load("./dataset/processed/test.npz")
    X_train, y_train = train_npz["X"], train_npz["y"]
    X_test, y_test   = test_npz["X"], test_npz["y"]

    # number of classes
    classes = open("./dataset/processed/classes.txt").read().splitlines()
    num_classes = len(classes)

    # train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_model((X_train, y_train), (X_test, y_test), num_classes, epochs=15, batch_size=64, lr=1e-3, device=device)
