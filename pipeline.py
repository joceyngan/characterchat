from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import logging
from pathlib import Path
import spacy
from pipeline.DialogueStateManager import DialogueStateManager
from pipeline.IntentClassifier import IntentClassifier
from pipeline.EmotionClassifier import EmotionClassifier
from pipeline.ResponseGenerator import ResponseGenerator
import warnings
import json
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def load_models(persona_json_path, device):
    pbar = tqdm(total=100, desc='Preparing chatbot')

    with open(persona_json_path, 'r') as file:
        data = json.load(file)
    persona = data.get('persona', [])
    pbar.update(5)

    intent_model_path = './models/intents/20230611214607-distilbert-base-uncased'
    ic = IntentClassifier(intent_model_path, device)
    pbar.update(25)

    emotion_model_path = './models/emotions/20230612054142-distilbert-base-uncased'
    ec = EmotionClassifier(emotion_model_path, device)
    pbar.update(30)

    gpt2_model_path = './models/gpt2/20230716193106-gpt2-medium'
    rg = ResponseGenerator(gpt2_model_path, device, persona)
    pbar.update(30)

    spacy_model = spacy.load('en_core_web_sm')
    pbar.update(10)
    pbar.close()
    print("Chatbot is ready.")

    return persona, ic, ec, rg, spacy_model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    persona_json_path = './data/persona/persona_mary.json'
    persona, ic, ec, rg, spacy_model = load_models(persona_json_path, device)

    dialogue_tracker = DialogueStateManager(max_conversations=5)
    remembered_entities = {}
    rounds_to_remember = 5

    dialogues = ''
    input_text = ''
    while input_text != 'exit':
        # Get user's input
        input_text = input("User : ")
        if input_text != 'exit' and len(input_text) > 0:
            user_input_str = f"user: {input_text} "
            dialogue_tracker.update_chat_history(user_input_str)

            # Get user's intent
            predicted_intent = ic.classify_intent(input_text)
            # print(f"Predicted intent: {predicted_intent}")

            # Get user's emotions
            top_emotion, top3_emotions = ec.classify_emotion(input_text)
            # emotions_str = ', '.join(
            # f"{emotion}: {confidence * 100:.2f}%" for emotion, confidence in top3_emotions)
            # print(f"Top emotion: {top_emotion}, Top 3 emotions: {emotions_str}")

            # Entity extraction and update
            doc = spacy_model(input_text)
            if doc.ents:
                # print("New entities found in this round:")
                for entity in doc.ents:
                    # print(f"\t{entity.text} ({entity.label_})")
                    # If the entity is new or forgotten, add it with age 0
                    remembered_entities[(entity.text, entity.label_)] = 0
            # else:
                # print("No new entities found in this round.")

            # Update ages and forget old entities
            entities_to_forget = []
            for entity, age in remembered_entities.items():
                remembered_entities[entity] = age + 1
                if remembered_entities[entity] > rounds_to_remember:
                    entities_to_forget.append(entity)

            # Forget old entities
            for entity in entities_to_forget:
                del remembered_entities[entity]

            # Update dialogue state tracker and chat history
            dialogue_tracker.update_state(conversation=input_text, intent=predicted_intent,
                                          emotion=top_emotion, entities=doc.ents)
            current_state = dialogue_tracker.get_state()
            chat_history = dialogue_tracker.get_chat_history()
            response = rg.generate_response_history(
                current_state, chat_history)

            dialogue_tracker.update_chat_history(f"bot:{response} ")
            print(f"Mary : {response}")

    # print("Chat history:\n ", dialogue_tracker.get_chat_history())
