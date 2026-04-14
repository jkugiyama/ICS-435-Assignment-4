import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments


df = pd.read_csv("data")
print(df.head())

jokes = df["Joke"].tolist()

print(jokes[:5])

dataset = Dataset.from_dict({"text": jokes})


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

MAX_LENGTH = 64

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH)

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])


model = GPT2LMHeadModel.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = False

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    dataloader_pin_memory=False,
    group_by_length=True,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()
model.config.use_cache = True
model.eval()

def generate_joke(prompt):
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    encoded_prompt = {
        key: value.to(model.device)
        for key, value in encoded_prompt.items()
    }

    output = model.generate(
        input_ids=encoded_prompt["input_ids"],
        attention_mask=encoded_prompt["attention_mask"],
        max_length=50,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(generate_joke("What did the"))