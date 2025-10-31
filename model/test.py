import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CNN Model (same as train.py) ----------------
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
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# ---------------- Evaluation ----------------
def evaluate(model, X_test, y_test, classes, device="cpu"):
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X_test)
        _, preds = torch.max(outputs, 1)

    y_true = y_test.cpu().numpy()
    y_pred = preds.cpu().numpy()

    # overall accuracy
    overall_acc = accuracy_score(y_true, y_pred)

    # classification report (includes precision, recall, f1, support)
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    print("\nClassification Report:\n")
    print(report)

    # per-class accuracy (recall already, but let’s show it explicitly)
    print("\nPer-Class Accuracy:")
    for i, cls in enumerate(classes):
        cls_mask = (y_true == i)
        cls_acc = (y_pred[cls_mask] == y_true[cls_mask]).mean()
        print(f"{cls:12s}: {cls_acc*100:.2f}%")

    print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ---------------- Main ----------------
if __name__ == "__main__":
    test_npz = np.load("./dataset/processed/test.npz")
    X_test, y_test = test_npz["X"], test_npz["y"]

    classes = open("./dataset/processed/classes.txt").read().splitlines()
    num_classes = len(classes)
    input_dim = X_test.shape[2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN1DFlightRegime(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))

    evaluate(model, X_test, y_test, classes, device)
