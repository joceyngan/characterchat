from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine


class DialogueStateManager:
    def __init__(self, max_conversations=3, repetition_threshold=80):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.max_conversations = max_conversations
        self.state = []
        self.chat_history = []
        self.recent_responses = []
        self.repetition_threshold = repetition_threshold

    def update_chat_history(self, conversation):
        self.chat_history.append(conversation)
        if len(self.chat_history) > self.max_conversations:
            self.chat_history.pop(0)

    def update_state(self, conversation, intent, emotion, entities):
        # Add new state information
        self.state.append({
            "conversation": conversation,
            "intent": intent,
            "emotion": emotion,
            "entities": entities
        })
        # Remove the oldest conversation if limit exceeded
        if len(self.state) > self.max_conversations:
            self.state.pop(0)

    def update_recent_responses(self, response):
        self.recent_responses.append(response)
        if len(self.recent_responses) > self.max_conversations:
            self.recent_responses.pop(0)

    def compute_similarity(self, current_response):
        if len(self.recent_responses) >= 1:
            print("checking similarity in response...")
            model_name = "bert-base-uncased"
            model = AutoModel.from_pretrained(model_name).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inputs1 = tokenizer(self.recent_responses[-1], return_tensors='pt',
                                truncation=True, padding=True).to(self.device)
            inputs2 = tokenizer(current_response, return_tensors='pt',
                                truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
                embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

            # Compute cosine similarity
            similarity = 1 - cosine(embeddings1, embeddings2)
            print("similarity", similarity)
            return similarity

    def get_state(self):
        return self.state

    def clear_state(self):
        self.state = []

    def get_chat_history(self):
        return self.chat_history

    def clear_chat_history(self):
        self.chat_history = []
