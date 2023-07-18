from transformers import GPT2Tokenizer, GPT2LMHeadModel, logging
import re
import warnings
import random

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


class ResponseGenerator:
    def __init__(self, model_path, device, persona):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_path).to(self.device)
        self.persona = persona

    def get_response_or_failsafe(self, response):
        failsafe_responses = [
            "I'm sorry, I didn't quite catch that. Could you please rephase?",
            "Maybe I am too old for this, can you explain more?",
            "I'm not sure I understand. Could you elaborate?",
            "Hmm, I dont think I quite get what do you mean.",
            "Apologies for the confusion. Will be great if you can explain more?",
            "I'm not sure about that, could you explain a bit more?",
            "Not quite sure, not an expert in this field.",
            "Hmm, I don't quite understand. Could you please clarify?",
            "I'm afraid I don't know much about this. Can you shed more light?",
            "Sorry, but I'm not familiar with that. I was a cardiologist.",
            "Oh, I'm not sure about that. Would you mind explaining it?",
            "This is outside my expertise, I'm more familiar with the medical field.",
            "I'm not quite sure what you mean. Could you clarify for me?",
            "I'm afraid I don't have an answer to that. Maybe we talk something else? Like the weather in Long Island?",
            "I'm not certain, my area of expertise is in cardiology.",
            "I don't quite understand. Can you explain it in another way?",
            "Hmm, that's new to me. Could you tell more?",
            "I'm sorry, I don't quite follow. Can you explain more?",
            "My apologies, but I didn't grasp your point. Could you elaborate?",
            "I'm not sure I understand. Could you clarify?",
            "That's beyond my knowledge. I'm more versed in health matters.",
            "Hmm, can you explain it a bit more? It's a bit out of my expertise.",
            "I don't quite get it, can you please elaborate?",
            "This is a bit out of my wheelhouse. Could you give more details?",
            "Sorry, I don't understand. Can you rephrase or explain it?"
        ]

        if response is None or response.strip() == "":
            return random.choice(failsafe_responses)
        else:
            return response

    def format_response(self, response):
        phrases_to_remove = ['user:', 'user:,', 'user: ,', 'user:!', 'user:.', 'user: .', 'user,',
                             'bot:', 'bot:,', 'bot: ,', 'bot:!', 'bot:.', 'bot: .', 'bot,',
                             'user:bot:', 'bot:bot:', 'bot:user:', 'user:user:',
                             'user: bot:', 'bot: bot:', 'bot: user:', 'user: user:']
        # Check if repeated sentences, remove if any
        sentences = re.split('(?<=[.!?]) +', response)
        stripped_sentences = [
            re.sub(r'^[.,!?]|$', '', s.lstrip()) for s in sentences]
        stripped_sentences = [re.sub(r'^\s+', '', s.lstrip())
                              for s in stripped_sentences]
        if len(stripped_sentences) >= 2 and stripped_sentences[0] == stripped_sentences[1]:
            shortened_response = sentences[0]
        else:
            # Randomly decide to return either 1 or 2 sentences
            num_sentences = random.randint(1, 2)
            shortened_response = ' '.join(
                [s.lstrip()for s in sentences[:num_sentences]])
        # Check if unecessary phrase, remove if any
        formatted_response = shortened_response
        for phrase in phrases_to_remove:
            formatted_response = formatted_response.replace(phrase, '', 1)
        # Check if unecessary char after shortening, remove if any
        formatted_response = re.sub(r'^[ ,!.:\[\]]', '', formatted_response)
        return formatted_response

    def generate_response_history(self, current_state, chat_history, max_length=256):
        state_info = ""
        for state in current_state[-1]:
            if 'emotion' in state and 'intent' in state:
                intent = re.sub(r'_', ' ', state['intent'])
                state_info += f"user emotion is {state['emotion']}, user wants to {intent}. "
            else:
                state_info += f""
        prompt = f"<persona> {self.persona} <USER> {chat_history} {state_info} bot: <BOT> "
        inputs = self.tokenizer.encode(
            prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1,
                                      temperature=0.9, top_k=2000, top_p=0.2)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(
            str(self.tokenizer.decode(inputs[0].tolist(), skip_special_tokens=True)), '', 1)
        response = self.get_response_or_failsafe(response)
        return self.format_response(response)
