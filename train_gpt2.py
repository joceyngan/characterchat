import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW, DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
from dataset import GPTDialogueDataset, GPTPersonaSDataset, GPTPersonaNoTagDataset
import math
from datetime import datetime
from pathlib import Path
from tqdm import trange
import wandb
import random


def append_to_log(path, message):
    with open(path, 'a') as f:
        f.write('\n'+message)
        f.close()


# setup env
datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
model_name = 'gpt2-medium'
model_name_str = model_name.replace('/', '-')
train_name = f'{datetime_str}-{model_name_str}'
# result_dir = Path('./models/gpt2_ft')
result_dir = Path('./models/gpt2_ft_no_tag')
model_save_path = result_dir/train_name
model_save_path.mkdir(exist_ok=True)
model_save_name = train_name+".pth"
print('model_save_path: ', model_save_path, ' created')
logs_save_path = model_save_path/'logs'
logs_save_path.mkdir(exist_ok=True)
print('logs_save_path: ', logs_save_path, ' created')
logs_save_path = logs_save_path/(train_name+".txt")
log_str = ''

# Load the dataset
train_ratio = 0.8
val_ratio = 0.10
test_ratio = 0.10
batch_size = 16
epochs = 50
learning_rate = 1e-5
weight_decay = 1e-4
num_warmup_steps = 0
data_path = './data/persona/persona_mary.json'
# ./data/nlg/gpt2_chat.json
# ./data/persona/persona_mary_emotion_intent.json
log_str += f"datetime: {datetime_str}\nmodel arch:{model_name}\nbatch size: {batch_size}\n"
log_str += f"train/val/test ratio: {train_ratio}/{val_ratio}/{test_ratio}\n"
log_str += f"epochs:{epochs}\n"
log_str += f"data_path:{data_path}\n"
append_to_log(logs_save_path, log_str)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {'additional_special_tokens':  [
    '<persona>', '<emotion>', '<intent>', '<USER>', '<BOT>']}
tokenizer.add_special_tokens(special_tokens_dict)
system_token_id = tokenizer.encode('<BOT>', return_tensors='pt')[0, 0]
append_to_log(logs_save_path, f"custom_tokens:{special_tokens_dict}\n")

# Configuration and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("device: ", device)
config = GPT2Config.from_pretrained(model_name)
config.pad_token_id = tokenizer.eos_token_id
model = GPT2LMHeadModel.from_pretrained(model_name, config=config).to(device)

# Resize token embeddings with additional tokens
model.resize_token_embeddings(len(tokenizer))

# Load dataset
# dataset = GPTDialogueDataset(data_path, tokenizer)
# dataset = GPTPersonaSDataset(data_path, tokenizer) # dataset with seperated emotion and intent tags
# dataset w/o emotion and intent tags
dataset = GPTPersonaNoTagDataset(data_path, tokenizer)


# Split the dataset
n_total = len(dataset)
train_size = round(train_ratio * n_total)
val_size = round(val_ratio * n_total)
test_size = n_total - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


def check_labels(dataloader, vocab_size):  # Functions for debugging purpose
    for i, batch in enumerate(dataloader):
        labels = batch['labels']
        max_label = labels.max().item()
        min_label = labels.min().item()
        if max_label >= vocab_size or min_label < -100:
            print(f"Batch {i}: Max label {max_label}, Min label {min_label}")
            print("First 10 labels:", labels.view(-1)[:10])
            input()


def inspect_data(dataloader):  # Functions for debugging purpose
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: {batch}")
        input()

# inspect_data(dataloader)
# check_labels(dataloader, 50257)


# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate,
                  weight_decay=weight_decay)
log_str = f"optimizer: AdamW\nlearning_rate: {learning_rate}\nweight_decay: {weight_decay}\n"
append_to_log(logs_save_path, log_str)

criterion = CrossEntropyLoss()

wandb.init(
    project="nlp-gpt",
    name=train_name,
    # track hyperparameters and run metadata
    config={
        "dataset": data_path.split('/')[-1],
        "optimizers": 'AdamW',
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }
)

# Scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=len(dataloader) * epochs
)
log_str = f"epochs: {epochs}\nnum_warmup_steps: {num_warmup_steps}\n"
append_to_log(logs_save_path, log_str)

epoch = 1
for epoch in trange(epochs, desc="Epoch"):
    print(f"Training...# of epochs: {epoch}")
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=inputs,
                        attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        wandb.log({"train loss": total_loss/len(train_dataloader)})

    log_str = f"Epoch {epoch}: Average Training Loss = {total_loss/len(train_dataloader)}\n"
    append_to_log(logs_save_path, log_str)
    print(f"Average Training Loss: {total_loss/len(train_dataloader)}")

    print(f"Evaluating...# of epochs: {epoch}")
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=inputs,
                            attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

        # Select a random sample from test dataset for generation
        random_sample = random.choice(test_dataset)
        input_ids = random_sample["input_ids"].to(device).unsqueeze(0)
        attention_mask = random_sample["attention_mask"].to(
            device).unsqueeze(0)

        # Split input_ids into user input and expected bot output
        bot_token_id = tokenizer.convert_tokens_to_ids('<BOT>')
        user_input_ids = input_ids[0][:input_ids[0].tolist().index(
            bot_token_id) + 1].unsqueeze(0)

        # Generate an output for the user input
        generated_ids = model.generate(user_input_ids, do_sample=True,
                                       temperature=0.9, max_length=100, pad_token_id=tokenizer.eos_token_id)

        # Extract expected output
        expected_output_ids = input_ids[0][input_ids[0].tolist().index(
            bot_token_id) + 1:]
        expected_output_text = tokenizer.decode(
            expected_output_ids, skip_special_tokens=True)

        generated_text = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True)
        user_input_text = tokenizer.decode(
            user_input_ids[0], skip_special_tokens=True)
        # generated_text = generated_text.replace(user_input_text, '', 1)
        print(f' Eval Input: {user_input_text}')
        append_to_log(logs_save_path, f' Eval Input: {user_input_text}\n')
        print(f' Gound Truth Response: {expected_output_text}')
        append_to_log(logs_save_path,
                      f' Gound Truth Response: {expected_output_text}\n')
        print(f' Eval Response: {generated_text}')
        append_to_log(logs_save_path, f' Eval Response: {generated_text}\n')

        val_perplexity = math.exp(total_loss / len(val_dataloader))
        print(f'Validation Perplexity: {val_perplexity}')
        wandb.log({"validation perplexity": val_perplexity})
        append_to_log(
            logs_save_path, f"Epoch {epoch}: Validation Perplexity = {val_perplexity}\n")
        print(f"Epoch {epoch} completed.")


# Save the model
torch.save(model.state_dict(), model_save_path/model_save_name)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print("model_save_path: ", model_save_path)
wandb.finish()
