import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments


df = pd.read_csv("data")
print(df.head())

jokes = df["Joke"].tolist()

print(jokes[:5])

dataset = Dataset.from_dict({"text": jokes})


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    encodings = tokenizer(example["text"], truncation=True, padding="max_length")
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

dataset = dataset.map(tokenize)


model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

def generate_joke(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        top_k=50
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_joke("What did the"))

generate_joke("Why did the")
generate_joke("What did the")
generate_joke("Don't you hate")