import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments


# Load dataset 
# Read the CSV containing 1,622 short jokes and extract the Joke column.
df = pd.read_csv("data")
print(df.head())

jokes = df["Joke"].tolist()  # Plain list of joke strings used throughout

print(jokes[:5])

# Wrap the list in a HuggingFace Dataset so it works with the Trainer API.
dataset = Dataset.from_dict({"text": jokes})


# Tokenizer setup 
# GPT-2 has no dedicated pad token; reuse EOS so the collator can pad
# variable-length sequences to the same length within each batch.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Limit sequence length to 64 tokens; most short jokes fit well within this
# window and it keeps each training step fast.
MAX_LENGTH = 64

def tokenize(example):
    # Truncate jokes longer than MAX_LENGTH and return raw token ids.
    # No padding here — the DataCollator handles dynamic padding per batch.
    return tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH)

# Apply tokenization in batches and drop the original text column since
# the Trainer only needs numerical token ids.
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])


# Model setup 
# Load the pre-trained GPT-2 weights as the starting point for fine-tuning.
model = GPT2LMHeadModel.from_pretrained("gpt2")
# Tell the model which token id represents padding so it ignores those positions.
model.config.pad_token_id = tokenizer.eos_token_id
# Disable the KV cache during training; it is only useful at inference time.
model.config.use_cache = False

# DataCollatorForLanguageModeling pads each batch to the longest sequence in
# that batch (dynamic padding) and automatically creates the labels tensor by
# copying input_ids, which is required for causal language-model loss.
# mlm=False selects causal (next-token prediction) mode, matching GPT-2.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments 
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,   # Larger batch → fewer steps, faster epoch
    num_train_epochs=3,              # Three full passes over the joke dataset
    logging_dir="./logs",
    dataloader_pin_memory=False,     # MPS (Apple Silicon) does not support pin_memory
    group_by_length=True,            # Batch similar-length sequences to minimise padding waste
    save_strategy="no",              # Skip checkpoint saves to speed up training
    report_to="none",                # Disable external loggers (wandb, tensorboard, etc.)
)

# Training 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()         # Fine-tune GPT-2 on the joke corpus
model.config.use_cache = True  # Re-enable KV cache to speed up autoregressive generation
model.eval()            # Switch off dropout for deterministic inference

# Inference 
def generate_joke(prompt):
    # Tokenize the prompt and move tensors to the same device as the model.
    # Passing attention_mask explicitly avoids a warning about pad/eos ambiguity.
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    encoded_prompt = {
        key: value.to(model.device)
        for key, value in encoded_prompt.items()
    }

    output = model.generate(
        input_ids=encoded_prompt["input_ids"],
        attention_mask=encoded_prompt["attention_mask"],
        max_new_tokens=60,    # Generate up to 60 new tokens beyond the prompt
        do_sample=True,       # Sample from the distribution rather than greedy decoding
        top_k=50,             # Restrict sampling to the 50 most likely next tokens
        top_p=0.95,           # Nucleus sampling: keep tokens covering 95% of probability mass
        temperature=0.9,      # Slightly below 1.0 to sharpen the distribution a little
        pad_token_id=tokenizer.eos_token_id,  # Silence pad_token warning during generation
    )

    # Decode the full sequence (prompt + generated tokens) back to a string.
    # skip_special_tokens removes EOS/PAD markers from the output.
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Evaluation set 1: dataset-based prompts 
# Extract the first three words of five actual jokes from the training set.
# This tests whether the model has learned joke structure for patterns it saw
# during fine-tuning.
dataset_prompts = [
    " ".join(jokes[i].split()[:3]) for i in [0, 2, 3, 7, 8]
]

print("\n=== EVALUATION SET 1: Dataset-based prompts ===")
for prompt in dataset_prompts:
    print(f"\nPrompt : {prompt!r}")
    print(f"Output : {generate_joke(prompt)}")

# Evaluation set 2: random prompts 
# Prompts whose three words do not appear at the start of any training joke.
# This tests generalization: can the model still produce joke-like output for
# out-of-distribution inputs?
random_prompts = [
    "Never trust a",
    "Scientists recently discovered",
    "My dog always",
    "The president decided",
    "Once upon a",
]

print("\n=== EVALUATION SET 2: Random prompts (not from dataset) ===")
for prompt in random_prompts:
    print(f"\nPrompt : {prompt!r}")
    print(f"Output : {generate_joke(prompt)}")