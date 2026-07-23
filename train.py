"""
train.py — Train a single-layer perceptron on the OR gate using PyTorch.

Outputs:
  • model_weights.pth      — learned weight tensor + bias
  • training_history.json  — per-epoch loss curve
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim


# ── Model ────────────────────────────────────────────────────────────────────
class Perceptron(nn.Module):
    """Single neuron: y = σ(w·x + b)"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


# ── Training ─────────────────────────────────────────────────────────────────
def train(
    epochs: int = 5000,
    lr: float = 0.1,
    seed: int = 42,
    model_path: str = "model_weights.pth",
    history_path: str = "training_history.json",
) -> None:
    torch.manual_seed(seed)

    # OR-gate truth table
    X = torch.tensor([[0.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 0.0],
                       [1.0, 1.0]])
    y = torch.tensor([[0.0],
                       [1.0],
                       [1.0],
                       [1.0]])

    model = Perceptron()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: list[float] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        history.append(loss_val)

        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:>5d}/{epochs}  Loss: {loss_val:.6f}")

    # ── Save artefacts ───────────────────────────────────────────────────
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model weights saved → {model_path}")

    with open(history_path, "w") as f:
        json.dump({"epochs": list(range(1, epochs + 1)), "loss": history}, f)
    print(f"✓ Training history saved → {history_path}")

    # ── Quick sanity check ───────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds = model(X)
    print("\n── OR-Gate Predictions ──")
    for inputs, pred, target in zip(X, preds, y):
        label = int(pred.item() >= 0.5)
        print(
            f"  {int(inputs[0].item())} OR {int(inputs[1].item())} "
            f"→ {pred.item():.4f} (class {label})  "
            f"{'✓' if label == int(target.item()) else '✗'}"
        )


if __name__ == "__main__":
    train()
