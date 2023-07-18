from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, logging
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
import time
import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


class IntentClassifier():
    def __init__(self, model_dir, device):
        self.device = device
        self.model_dir = model_dir
        self.model_path = Path(model_dir)
        self.train_name = self.model_path.name
        timestamp = self.train_name.split('-')[0]
        split_string = self.train_name.split(timestamp + '-')
        self.model_arch = split_string[1] if len(split_string) > 1 else ''
        with open(model_dir+'/label_encoder.pkl', 'rb') as f:
            self.intent_label_encoder = pickle.load(f)

        # Define the model architecture
        self.intent_model = DistilBertForSequenceClassification.from_pretrained(
            self.model_arch, num_labels=len(self.intent_label_encoder.classes_))
        self.intent_model_path = self.model_path/(self.train_name + ".pth")
        self.intent_model.load_state_dict(torch.load(
            self.intent_model_path, map_location=device))
        self.intent_model.to(device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_arch)

    def classify_intent(self, sentence):
        # Tokenize the input
        inputs = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, padding=True)
        inputs = {name: tensor.to(self.device)
                  for name, tensor in inputs.items()}
        with torch.no_grad():
            outputs = self.intent_model(**inputs)
        # Get predicted class
        predicted_class_idx = np.argmax(
            outputs.logits.cpu().numpy(), axis=1)[0]
        predicted_intent = self.intent_label_encoder.inverse_transform(
            [predicted_class_idx])[0]

        return predicted_intent
