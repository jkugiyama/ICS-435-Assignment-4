# Standard library imports
from pathlib import Path
import random

# Third-party imports for visualization, data processing, and deep learning
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# ── Global hyperparameters ──────────────────────────────────────────────────
SEED = 42            # Fixed seed for reproducibility across all RNGs
BATCH_SIZE = 64      # Number of samples per training/validation/test step
EPOCHS = 5           # Total number of passes through the training set
LEARNING_RATE = 1e-3 # Adam optimizer step size
VALIDATION_SPLIT = 0.2  # Fraction of training data reserved for validation


def set_seed(seed: int) -> None:
    # Seed every RNG used by Python, NumPy, and PyTorch so results are
    # reproducible across independent runs.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TransformSubset(Dataset):
    """
    A lightweight Dataset wrapper that applies a transform to a subset of
    another dataset identified by a list of indices.

    This allows the training and validation splits to have different transforms
    (e.g., augmentation only on training) while sharing the same base dataset.
    """

    def __init__(self, dataset: Dataset, indices: list[int], transform=None):
        self.dataset = dataset    # The full underlying dataset
        self.indices = indices    # Indices into the full dataset for this split
        self.transform = transform  # Optional transform applied per sample

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        # Map the local index to the corresponding sample in the full dataset
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ── Model 1: Baseline CNN ───────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    """
    Two-layer CNN baseline.

    Architecture:
        Conv(1→32, k=3) → ReLU → MaxPool(2x2)
        Conv(32→64, k=3) → ReLU → MaxPool(2x2)
        FC(64*5*5 → 128) → ReLU → FC(128 → 10)

    Input:  (N, 1, 28, 28) grayscale images
    Output: (N, 10) raw class logits
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # 28x28 → 26x26, 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 13x13 → 11x11, 64 filters
        self.pool = nn.MaxPool2d(2, 2)                 # Halves spatial dimensions
        self.fc1 = nn.Linear(64 * 5 * 5, 128)         # Flattened feature map → 128
        self.fc2 = nn.Linear(128, 10)                  # 128 → 10 class scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))  # (N, 32, 13, 13)
        x = self.pool(torch.relu(self.conv2(x)))  # (N, 64, 5, 5)
        x = x.view(x.size(0), -1)                # Flatten to (N, 1600)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)                           # Raw logits; softmax applied by loss
        return x


# ── Model 2: Baseline + Batch Normalization + Dropout ───────────────────────
class BatchNormDropoutCNN(nn.Module):
    """
    Modification 1: adds BatchNorm after each conv layer and Dropout before
    the output layer.

    BatchNorm normalizes activations per mini-batch, stabilizing training and
    allowing higher learning rates.  Dropout randomly zeroes 50% of hidden
    units during training, acting as a regularizer to reduce overfitting.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)              # Normalize after first conv
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)              # Normalize after second conv
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)             # 50% dropout for regularization
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BN is applied after conv and before ReLU (conv → BN → ReLU → pool)
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # (N, 32, 13, 13)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # (N, 64, 5, 5)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))  # Dropout only during training
        x = self.fc2(x)
        return x


# ── Model 3: Deeper CNN + Data Augmentation ─────────────────────────────────
class DeeperCNN(nn.Module):
    """
    Modification 2: four convolutional layers (two blocks of two), batch norm,
    dropout, and trained with data augmentation.

    Using 'same' padding keeps spatial dimensions stable within each block so
    only MaxPool reduces the feature map size.  The deeper feature extractor
    captures more complex patterns, while augmentation (flip + rotation)
    improves generalization to unseen orientations.
    """

    def __init__(self):
        super().__init__()
        # Feature extractor: two convolutional blocks, each ending with MaxPool
        self.features = nn.Sequential(
            # Block 1: 28x28 → 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # keeps spatial size
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 28x28 → 14x14
            # Block 2: 14x14 → 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # 14x14 → 7x7
        )
        # Classifier head: flatten → dense → dropout → 10 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),  # 64 channels * 7 * 7 spatial
            nn.ReLU(),
            nn.Dropout(0.4),             # 40% dropout
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)     # Extract spatial features
        x = self.classifier(x)   # Classify based on extracted features
        return x


def make_data_root() -> Path:
    # Resolve the path relative to this script so it works from any working
    # directory.  Guard against a plain file named 'datasets' that would
    # prevent torchvision from creating the required subdirectories.
    project_dir = Path(__file__).resolve().parent
    data_root = project_dir / "datasets"
    if data_root.exists() and not data_root.is_dir():
        raise RuntimeError(
            f"Expected a directory for dataset storage but found a file at {data_root}."
        )
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


def get_split_indices(dataset_len: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    # Generate a reproducible random permutation and split into val/train
    # indices.  Using a dedicated Generator keeps this split independent from
    # other random operations in the same run.
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_len, generator=generator).tolist()
    val_size = int(dataset_len * val_split)
    val_indices = indices[:val_size]     # First val_size indices → validation
    train_indices = indices[val_size:]   # Remaining indices → training
    return train_indices, val_indices


def build_loaders(data_root: Path, train_transform, val_test_transform):
    # Download the full 60,000-sample training split without any transform;
    # transforms are applied per-sample via TransformSubset so train and val
    # can have different augmentation pipelines.
    base_train = torchvision.datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        download=True,
    )
    train_indices, val_indices = get_split_indices(len(base_train), VALIDATION_SPLIT, SEED)

    # Wrap each split with its own transform pipeline
    train_dataset = TransformSubset(base_train, train_indices, transform=train_transform)
    val_dataset = TransformSubset(base_train, val_indices, transform=val_test_transform)

    # The official 10,000-sample test split uses the same transform as validation
    # (no augmentation) for an unbiased final evaluation.
    test_dataset = torchvision.datasets.FashionMNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=val_test_transform,
    )

    # shuffle=True only for training to prevent the model learning batch order
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def train_and_validate(model, train_loader, val_loader, device):
    # CrossEntropyLoss combines LogSoftmax + NLLLoss; suitable for multi-class
    # classification because it outputs raw logits from the model.
    criterion = nn.CrossEntropyLoss()
    # Adam adapts the learning rate per parameter, typically converging faster
    # than plain SGD for this type of task.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []  # Per-epoch mean training loss
    val_losses = []    # Per-epoch mean validation loss

    for epoch in range(EPOCHS):
        # ── Training phase ──────────────────────────────────────────────────
        model.train()  # Enables dropout and batch norm training behaviour
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()          # Clear gradients from previous step
            outputs = model(images)         # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()                 # Compute gradients via backprop
            optimizer.step()               # Update model weights

            running_train_loss += loss.item()

        mean_train_loss = running_train_loss / len(train_loader)
        train_losses.append(mean_train_loss)

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()  # Disables dropout; uses running stats for batch norm
        running_val_loss = 0.0
        with torch.no_grad():  # Disable gradient tracking to save memory
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
    # Collect all predictions and ground-truth labels in a single pass
    # over the test set before computing any metrics.
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            # Take the class index with the highest logit as the prediction
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds_array = np.array(all_preds)
    all_labels_array = np.array(all_labels)

    # Accuracy: fraction of samples correctly classified
    test_accuracy = float((all_preds_array == all_labels_array).mean())
    # Weighted averages weight each class by its support (number of true samples)
    precision = precision_score(all_labels_array, all_preds_array, average="weighted", zero_division=0)
    recall = recall_score(all_labels_array, all_preds_array, average="weighted", zero_division=0)
    f1 = f1_score(all_labels_array, all_preds_array, average="weighted", zero_division=0)
    # Confusion matrix rows = true class, columns = predicted class
    cm = confusion_matrix(all_labels_array, all_preds_array)
    return test_accuracy, precision, recall, f1, cm


def save_loss_plot(train_losses, val_losses, model_name: str, output_dir: Path):
    # Plot train vs. validation loss per epoch to visualize learning progress
    # and catch signs of overfitting (val loss rising while train loss falls).
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Loss Curves - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / f"loss_{model_name}.png"
    plt.savefig(output_path)  # Save to disk; plt.show() is intentionally omitted
    plt.close()               # Release memory


def save_confusion_matrix(cm: np.ndarray, model_name: str, output_dir: Path):
    # A confusion matrix (10x10 for FashionMNIST) shows where the model
    # confuses one class for another; diagonal cells are correct predictions.
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    ticks = np.arange(10)  # One tick per class (0-9)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()
    output_path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path)
    plt.close()


def run_experiment(model_name: str, model, train_transform, val_test_transform, data_root: Path, output_dir: Path, device):
    # Orchestrates a full experiment cycle: data loading, training,
    # evaluation, and artifact saving for a single model configuration.
    print(f"\n===== {model_name} =====")
    train_loader, val_loader, test_loader = build_loaders(
        data_root,
        train_transform,
        val_test_transform,
    )

    model = model.to(device)  # Move model parameters to CPU or GPU
    train_losses, val_losses = train_and_validate(model, train_loader, val_loader, device)
    test_accuracy, precision, recall, f1, cm = evaluate(model, test_loader, device)

    # Save plots to the results directory for inclusion in the report
    save_loss_plot(train_losses, val_losses, model_name, output_dir)
    save_confusion_matrix(cm, model_name, output_dir)

    print(f"Test Accuracy ({model_name}): {test_accuracy:.4f}")
    print(f"Precision ({model_name}): {precision:.4f}")
    print(f"Recall ({model_name}): {recall:.4f}")
    print(f"F1-Score ({model_name}): {f1:.4f}")
    print(f"Confusion Matrix ({model_name}):\n{cm}")

    # Return a result dict so the caller can compare across experiments
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
    # Print a side-by-side metric table and show how much each model improved
    # over the baseline to make the ablation study easy to read.
    print("\n===== Comparison Summary =====")
    baseline = results[0]  # First experiment is always the baseline
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
    # ── Setup ────────────────────────────────────────────────────────────────
    set_seed(SEED)  # Ensure reproducibility before any data loading or model init

    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = make_data_root()  # Prepare the dataset download directory
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure results folder exists

    # ── Transforms ───────────────────────────────────────────────────────────
    # Base transform: only convert PIL image to a [0,1] float tensor
    base_transform = transforms.Compose([transforms.ToTensor()])

    # Augmented transform: random flip and small rotation increase training
    # diversity, then convert to tensor (used only during training)
    augmented_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),  # Mirror image horizontally
            transforms.RandomRotation(10),            # Rotate up to ±10 degrees
            transforms.ToTensor(),
        ]
    )

    # ── Experiment registry ──────────────────────────────────────────────────
    # Each entry is (name, model instance, train transform, eval transform).
    # Val and test always use base_transform to avoid augmentation leaking into
    # evaluation.
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
            augmented_transform,  # Augmented training data
            base_transform,       # Clean eval data
        ),
    ]

    # ── Run all experiments ──────────────────────────────────────────────────
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

    # Print final side-by-side comparison across all three models
    print_comparison_summary(results)


if __name__ == "__main__":
    main()