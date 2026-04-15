# ICS-435-Assignment-4

This assignment contains two independent experiments: a CNN image classifier trained on FashionMNIST, and a GPT-2 fine-tuned joke generator.

---

## Part 1: CNN Image Classification (`cnn_model.py`)

### Overview
Three CNN architectures are trained and compared on the FashionMNIST dataset. The goal is to evaluate how architectural choices (batch normalization, dropout, depth, data augmentation) affect classification performance.

### Dataset
Dataset: FashionMNIST (via torchvision)

Samples: 60,000 training / 10,000 test

Input: 28×28 grayscale images

Classes: 10 clothing categories

The training set is further split 80/20 into train and validation subsets.

### Models Implemented

- **Baseline CNN**: Two conv layers with max-pooling and fully connected layers
- **BatchNorm + Dropout CNN**: Same as baseline with batch normalization and 0.5 dropout
- **Deeper CNN**: Four conv layers with batch normalization, max-pooling, and data augmentation (random horizontal flip, random rotation)

### Evaluation Metrics

Each model is evaluated using: Accuracy, Precision (weighted), Recall (weighted), F1-score (weighted), Confusion Matrix

### Outputs

Loss curve plots and confusion matrix images are saved to `./results/`.

### How to Run

Make sure required packages are installed:
```
pip install torch torchvision numpy matplotlib scikit-learn
```

Then run:
```
python cnn_model.py
```

Results and plots will be saved to `./results/`.

---

## Part 2: GPT-2 Joke Generator (`joke_generation.py`)

### Overview
A pre-trained GPT-2 model is fine-tuned on a dataset of jokes to generate new jokes given a short text prompt.

### Dataset
A CSV file containing a `Joke` column. The file path is set via `pd.read_csv("data")` — update this to match your local CSV path.

### Training
- Model: `gpt2` (from Hugging Face)
- Max token length: 64
- Batch size: 8
- Epochs: 3
- Checkpoints saved to `./results/`

### Evaluation
Two sets of prompts are used after training:

- **Set 1**: First three words taken from actual jokes in the dataset
- **Set 2**: Five arbitrary prompts not drawn from the dataset

Generated outputs are printed to the terminal.

### How to Run

Make sure required packages are installed:
```
pip install pandas datasets transformers torch
```

Update the CSV path in the script if needed, then run:
```
python joke_generation.py
```

Generated jokes will print in the terminal.
