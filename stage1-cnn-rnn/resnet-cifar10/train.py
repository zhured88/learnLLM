"""训练循环 — ResNet CIFAR-10"""

import time
import torch
import torch.nn as nn


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    device: torch.device = None,
    lr_milestones: tuple = (25, 40),
    label: str = "",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9,
        weight_decay=weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=0.1,
    )

    print(f"\n{'='*60}")
    print(f"  {label or 'Training'}")
    print(f"  Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Test Loss':>12} {'Test Acc':>10} {'LR':>8} {'Time':>8}")
    print("-" * 72)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"{epoch:>5} {train_loss:>12.4f} {train_acc:>9.2%} "
              f"{test_loss:>12.4f} {test_acc:>9.2%} {current_lr:>8.1e} {elapsed:>7.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc

    print(f"\nBest Test Accuracy: {best_acc:.2%}")
    return best_acc
