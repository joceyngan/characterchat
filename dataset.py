import torch
from torch.utils.data import Dataset
import json

class PersonaDataset(Dataset):
    def __init__(self, encoded_personas):
        self.encoded_personas = encoded_personas

    def __len__(self):
        return len(self.encoded_personas)

    def __getitem__(self, idx):
        return self.encoded_personas[idx]["input_ids"], self.encoded_personas[idx]["attention_mask"]

class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["data"]
        label = self.data.iloc[idx]["encoded_labels"]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EmotionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["data"]
        label = self.data.iloc[idx]["encoded_labels"]
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class GPTDialogueDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        with open(file_path, 'r') as file:
            self.data = json.load(file)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # format: "A: <content> [Intent: <intents>] [Emotion: <emotions>] [Entities: <entities>]
        #          B: <content> [Intent: <intents>] [Emotion: <emotions>] [Entities: <entities>]"
        input_text = ""

        speaker = entry['speaker']
        text = entry['content']
        intents = ', '.join(entry.get('intents', []))
        emotions = ', '.join(entry.get('emotions', []))
        entities = ', '.join([f"{e['type']}:{e['value']}" for e in entry.get('entities', []) if 'type' in e and 'value' in e])

        input_text += f"{speaker}: {text} [Intent: {intents}] [Emotion: {emotions}] [Entities: {entities}] "

        # Tokenize the texts
        tokenized = self.tokenizer.encode_plus(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": tokenized["input_ids"].squeeze(), "attention_mask": tokenized["attention_mask"].squeeze()}