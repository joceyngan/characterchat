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


class EmotionClassifier():
    def __init__(self, model_dir, device):
        self.device = device
        self.model_dir = model_dir
        self.model_path = Path(model_dir)
        self.train_name = self.model_path.name
        timestamp = self.train_name.split('-')[0]
        split_string = self.train_name.split(timestamp + '-')
        self.model_arch = split_string[1] if len(split_string) > 1 else ''
        with open(model_dir+'/label_encoder.pkl', 'rb') as f:
            self.emotion_label_encoder = pickle.load(f)

        # Define the model architecture
        self.emotion_model = DistilBertForSequenceClassification.from_pretrained(
            self.model_arch, num_labels=len(self.emotion_label_encoder.classes_))
        self.emotion_model_path = self.model_path/(self.train_name + ".pth")
        self.emotion_model.load_state_dict(torch.load(
            self.emotion_model_path, map_location=device))
        self.emotion_model.to(device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_arch)

    def classify_emotion(self, sentence):
        inputs = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, padding=True)
        inputs = {name: tensor.to(self.device)
                  for name, tensor in inputs.items()}
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
        # Convert logits to probabilities
        probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_emotions = [(self.emotion_label_encoder.inverse_transform(
            [idx])[0], probabilities[idx]) for idx in top_indices]
        # Get predicted class
        predicted_class_idx = np.argmax(
            outputs.logits.cpu().numpy(), axis=1)[0]
        predicted_emotion = self.emotion_label_encoder.inverse_transform(
            [predicted_class_idx])[0]

        return predicted_emotion, top_emotions

    def classify_emotion_batch(self, sentences):
        encodings = self.tokenizer.batch_encode_plus(
            sentences, truncation=True, padding=True, return_tensors='pt')
        encodings = {name: tensor.to(self.device)
                     for name, tensor in encodings.items()}
        with torch.no_grad():
            outputs = self.emotion_model(**encodings)
            logits = outputs.logits

        # Convert logits to probabilities
        probabilities_batch = F.softmax(logits, dim=1).cpu().numpy()
        top_indices_batch = np.argsort(probabilities_batch, axis=1)[
            :, -3:][:, ::-1]
        top_emotions_batch = [[(self.emotion_label_encoder.inverse_transform([idx])[0], probabilities_batch[i, idx])
                               for idx in top_indices] for i, top_indices in enumerate(top_indices_batch)]

        # Get predicted class for each input
        predicted_class_indices = np.argmax(logits.cpu().numpy(), axis=1)
        predicted_emotions = [self.emotion_label_encoder.inverse_transform([idx])[0] for idx in predicted_class_indices]

        return predicted_emotions, top_emotions_batch
