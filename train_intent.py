import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from dataset import IntentDataset
from datetime import datetime
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# setup env
datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
model_name = 'distilbert-base-uncased'
train_name = f'{datetime_str}-{model_name}'
result_dir = Path('./models/intents')
model_save_path = result_dir/train_name
model_save_path.mkdir(exist_ok=True)
model_save_name = train_name+".pth"
print('model_save_path: ', model_save_path, ' created')
logs_save_path = model_save_path/'logs'
logs_save_path.mkdir(exist_ok=True)
print('logs_save_path: ', logs_save_path, ' created')

# Load the dataset
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
batch_size = 16
data_path = './data/nlu/intents/intents_data.csv'
dataset = pd.read_csv(data_path)

# Label Encoding
label_encoder = LabelEncoder()
dataset["encoded_labels"] = label_encoder.fit_transform(dataset["label"])

# Train Test Split
train_data, temp_data = train_test_split(dataset, test_size=(val_ratio + test_ratio), random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
print("Train dataset size:", len(train_data))
print("Validation dataset size:", len(val_data))
print("Test dataset size:", len(test_data))

# Load DistilBert Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# DataLoaders
train_dataset = IntentDataset(train_data, tokenizer)
val_dataset = IntentDataset(val_data, tokenizer)
test_dataset = IntentDataset(test_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Load DistilBert Model
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

wandb.init(
    project="nlp-intent",
    name=train_name,
    # track hyperparameters and run metadata
    config={
    "dataset": data_path.split('/')[-1],
    "optimizers": 'AdamW',
    }
)

# Training
training_args = TrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=1000,
    logging_dir=logs_save_path,
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy='steps',
    learning_rate=1e-5,
    weight_decay=0.001,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    report_to="wandb"
)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Start training
train_result = trainer.train()
eval_results = trainer.evaluate()

# Save the model
torch.save(model.state_dict(), model_save_path/model_save_name)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
with open(model_save_path/'label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Test the model and get predictions
test_preds = trainer.predict(test_dataset)
test_logits = test_preds.predictions
test_preds_labels = np.argmax(test_logits, axis=-1)

# Compute evaluation metrics
test_labels = label_encoder.transform(test_data['label'].values)
test_metrics = compute_metrics((test_logits, test_labels))
for key, value in test_metrics.items():
    print(f"{key}: {value:.4f}")

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, title=train_name+' Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    # Normalize the confusion matrix to get the accuracy per class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm_normalized = pd.DataFrame(cm_normalized, index=classes, columns=classes)

    # Custom cell label format to display count and accuracy
    cell_labels = np.array([["{}\n{:.1%}".format(count, acc) for count, acc in zip(row_counts, row_accs)]
                            for row_counts, row_accs in zip(cm, cm_normalized)])

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df_cm, annot=cell_labels, cmap="RdPu", fmt='', ax=ax,
                cbar=False, xticklabels=classes, yticklabels=classes)
    # plt.title(title, y=1.08)
    # plt.text(0.5, 1.02, "Top-1 Acc: {:.2%} | Top-2 Acc: {:.2%} | Top-3 Acc: {:.2%}".format(top1_acc, top2_acc, top3_acc),
    #          horizontalalignment='center',
    #          fontsize=12,
    #          transform=plt.gca().transAxes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    metrics_str = ', '.join(
        [f"{key}: {value:.4f}" for key, value in test_metrics.items()])
    ax.set_title(f"{title}\n{metrics_str}")
    # add the evaluation metrics as text within the plot
    # metrics_str = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
    # ax.text(0.95, 0.5, metrics_str, fontsize=12, ha='right', va='center', transform=ax.transAxes)
    plt.savefig(model_save_path/f"{train_name}_cm.png")
    # plt.show()

label_names = label_encoder.classes_
unique_labels = sorted(list(set(test_labels)))
plot_confusion_matrix(test_labels, test_preds_labels, classes=label_names)