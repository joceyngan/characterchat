import json
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens_dict = {'additional_special_tokens':  [
    '<persona>', '<emotion>', '<intent>', '<USER>', '<BOT>']}
tokenizer.add_special_tokens(special_tokens_dict)
custom_tokens_list = ['<emotion>', '<intent>', '<USER>', '<SYSTEM>']
tokenizer.add_tokens(custom_tokens_list)

# file_path = './data/persona/persona_mary_emotion_intent.json'
file_path = './data/persona/persona_mary.json'
with open(file_path, 'r') as file:
    data = json.load(file)

defined_max_length = 256  # Define the maximum length
num_dialogues = 0
num_system_token_dialogues = 0
num_exceed_max_len_dialogues = 0
max_len_dialogue = 0

persona = data.get('persona', [])
for dialogue in data.get('dialogues', []):
    # input_text = f"<persona> {persona} <USER> {dialogue['input']} <emotion> {dialogue['emotion']} <intent> {dialogue['intent']} <BOT> "
    input_text = f"<persona> {persona} <USER> user: {dialogue['input']} bot: <BOT>"
    response_text = f"{dialogue['response']}"
    dialogue_text = input_text + tokenizer.eos_token + response_text

    tokenized_dialogue = tokenizer.encode(dialogue_text)

    if '<BOT>' in dialogue_text:
        num_system_token_dialogues += 1

    dialogue_len = len(tokenized_dialogue)
    if dialogue_len > defined_max_length:
        num_exceed_max_len_dialogues += 1

    # Check the maximum length
    max_len_dialogue = max(max_len_dialogue, dialogue_len)
    # Count the total number of dialogues
    num_dialogues += 1

print(f'Total number of dialogues: {num_dialogues}')
print(
    f'Number of dialogues containing <BOT> token: {num_system_token_dialogues}')
print(
    f'Number of dialogues exceeding {defined_max_length} tokens: {num_exceed_max_len_dialogues}')
print(f'Maximum length of dialogues: {max_len_dialogue} tokens')
