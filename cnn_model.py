from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TransformSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BatchNormDropoutCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_data_root() -> Path:
    project_dir = Path(__file__).resolve().parent
    data_root = project_dir / "datasets"
    if data_root.exists() and not data_root.is_dir():
        raise RuntimeError(
            f"Expected a directory for dataset storage but found a file at {data_root}."
        )
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


def get_split_indices(dataset_len: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_len, generator=generator).tolist()
    val_size = int(dataset_len * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def build_loaders(data_root: Path, train_transform, val_test_transform):
    base_train = torchvision.datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        download=True,
    )
    train_indices, val_indices = get_split_indices(len(base_train), VALIDATION_SPLIT, SEED)

    train_dataset = TransformSubset(base_train, train_indices, transform=train_transform)
    val_dataset = TransformSubset(base_train, val_indices, transform=val_test_transform)

    test_dataset = torchvision.datasets.FashionMNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=val_test_transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def train_and_validate(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        mean_train_loss = running_train_loss / len(train_loader)
        train_losses.append(mean_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        mean_val_loss = running_val_loss / len(val_loader)
        val_losses.append(mean_val_loss)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {mean_train_loss:.4f} | Val Loss: {mean_val_loss:.4f}"
        )

    return train_losses, val_losses


def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds_array = np.array(all_preds)
    all_labels_array = np.array(all_labels)
    test_accuracy = float((all_preds_array == all_labels_array).mean())
    precision = precision_score(all_labels_array, all_preds_array, average="weighted", zero_division=0)
    recall = recall_score(all_labels_array, all_preds_array, average="weighted", zero_division=0)
    f1 = f1_score(all_labels_array, all_preds_array, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels_array, all_preds_array)
    return test_accuracy, precision, recall, f1, cm


def save_loss_plot(train_losses, val_losses, model_name: str, output_dir: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Loss Curves - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / f"loss_{model_name}.png"
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: Path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    ticks = np.arange(10)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()
    output_path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path)
    plt.close()


def run_experiment(model_name: str, model, train_transform, val_test_transform, data_root: Path, output_dir: Path, device):
    print(f"\n===== {model_name} =====")
    train_loader, val_loader, test_loader = build_loaders(
        data_root,
        train_transform,
        val_test_transform,
    )

    model = model.to(device)
    train_losses, val_losses = train_and_validate(model, train_loader, val_loader, device)
    test_accuracy, precision, recall, f1, cm = evaluate(model, test_loader, device)

    save_loss_plot(train_losses, val_losses, model_name, output_dir)
    save_confusion_matrix(cm, model_name, output_dir)

    print(f"Test Accuracy ({model_name}): {test_accuracy:.4f}")
    print(f"Precision ({model_name}): {precision:.4f}")
    print(f"Recall ({model_name}): {recall:.4f}")
    print(f"F1-Score ({model_name}): {f1:.4f}")
    print(f"Confusion Matrix ({model_name}):\n{cm}")

    return {
        "name": model_name,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def print_comparison_summary(results):
    print("\n===== Comparison Summary =====")
    baseline = results[0]
    for result in results:
        improvement = result["test_accuracy"] - baseline["test_accuracy"]
        print(
            f"{result['name']}: "
            f"accuracy={result['test_accuracy']:.4f}, "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"f1={result['f1']:.4f}, "
            f"delta_vs_baseline={improvement:+.4f}"
        )


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = make_data_root()
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_transform = transforms.Compose([transforms.ToTensor()])
    augmented_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    experiments = [
        (
            "baseline",
            BaselineCNN(),
            base_transform,
            base_transform,
        ),
        (
            "batchnorm_dropout",
            BatchNormDropoutCNN(),
            base_transform,
            base_transform,
        ),
        (
            "deeper_augmented",
            DeeperCNN(),
            augmented_transform,
            base_transform,
        ),
    ]

    results = []
    for name, model, train_transform, eval_transform in experiments:
        result = run_experiment(
            model_name=name,
            model=model,
            train_transform=train_transform,
            val_test_transform=eval_transform,
            data_root=data_root,
            output_dir=output_dir,
            device=device,
        )
        results.append(result)

    print_comparison_summary(results)


if __name__ == "__main__":
    main()