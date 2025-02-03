import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

import random
import json
import os
import tkinter as tk
from tkinter import scrolledtext
from spacy.tokens import Doc
from textblob import TextBlob

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')

# Try to add SpacyTextBlob, but manually register polarity if needed
try:
    spacy_text_blob = SpacyTextBlob(nlp)
    if "spacytextblob" not in nlp.pipe_names:
        nlp.add_pipe("spacytextblob", last=True)
except Exception as e:
    print("Warning: SpacyTextBlob failed to load.", e)

# Manually register polarity extension if missing
if not Doc.has_extension("polarity"):
    Doc.set_extension("polarity", getter=lambda doc: TextBlob(doc.text).sentiment.polarity, force=True)

class Chatbot:
    def __init__(self):
        self.memory = []
        self.knowledge_base = self.load_knowledge_base()
        self.nlp = nlp  # Use the globally loaded spaCy model

    def load_knowledge_base(self):
        if os.path.isfile('knowledge_base.json'):
            with open('knowledge_base.json', 'r') as file:
                return json.load(file)
        else:
            return {
                'python': 'Python is an excellent programming language for beginners and experts alike.',
                'weather': 'Weather can really affect our mood, don’t you think?',
                'music': 'Music is a universal language that brings people together.',
            }

    def save_knowledge_base(self):
        with open('knowledge_base.json', 'w') as file:
            json.dump(self.knowledge_base, file)

    def tokenize_input(self, text):
        doc = self.nlp(text)
        return [token.text for token in doc]

    def get_response(self, user_input):
        self.memory.append(user_input)
        user_input_lower = user_input.lower()
        learn_response = self.learn_new_fact(user_input_lower)
        if learn_response:
            return learn_response
        sentiment = self.analyze_sentiment(user_input)
        tokens = self.tokenize_input(user_input_lower)
        knowledge_response = self.check_knowledge(tokens)
        if knowledge_response:
            return knowledge_response
        return self.generate_response(sentiment)

    def learn_new_fact(self, user_input):
        if user_input.startswith('remember that'):
            fact = user_input.replace('remember that', '').strip()
            tokens = self.tokenize_input(fact)
            if tokens:
                key = tokens[0]
                self.knowledge_base[key] = fact
                self.save_knowledge_base()
                return f"Got it! I'll remember that {fact}."
            else:
                return "Hmm, I didn't catch that. Could you tell me the fact again?"
        return None

    def analyze_sentiment(self, text):
        doc = self.nlp(text)
        sentiment = doc._.polarity  # Now always works due to manual polarity registration
        return {'compound': sentiment}

    def check_knowledge(self, tokens):
        for word in tokens:
            if word in self.knowledge_base:
                return self.knowledge_base[word]
        for kb_word in self.knowledge_base:
            token1 = self.nlp(word)[0]
            token2 = self.nlp(kb_word)[0]
            similarity = token1.similarity(token2)
            if similarity > 0.8:
                return self.knowledge_base[kb_word]
        return None

    def generate_response(self, sentiment):
        if self.memory:
            last_input = self.memory[-1]
            response = f"Earlier you mentioned '{last_input}'. Let's talk more about that."
        else:
            response = random.choice([
                "Tell me more!", "That's interesting.", "How does that make you feel?", "Why do you think that is?"
            ])
        if sentiment['compound'] > 0.5:
            response += " You seem happy!"
        elif sentiment['compound'] < -0.5:
            response += " I'm sorry to hear that you're feeling this way."
        return response

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot")
        self.chatbot = Chatbot()
        self.chat_display = scrolledtext.ScrolledText(master, state='disabled', wrap='word')
        self.chat_display.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.input_frame = tk.Frame(master)
        self.input_frame.pack(pady=5)
        self.input_field = tk.Entry(self.input_frame, width=80)
        self.input_field.pack(side=tk.LEFT, padx=5)
        self.input_field.bind("<Return>", self.send_message)
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.LEFT)
        self.display_message("Chatbot: Hello! How can I assist you today?")

    def display_message(self, message):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, message + '\n\n')
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def send_message(self, event=None):
        user_input = self.input_field.get()
        self.input_field.delete(0, tk.END)
        if user_input.strip():
            self.display_message(f"You: {user_input}")
            if user_input.lower() in ['bye', 'exit', 'quit']:
                self.display_message("Chatbot: Goodbye! It was nice chatting with you.")
                self.master.after(2000, self.master.quit)
            else:
                response = self.chatbot.get_response(user_input)
                self.display_message(f"Chatbot: {response}")
        else:
            self.display_message("Chatbot: Please enter a message so I can respond.")

def main():
    root = tk.Tk()
    gui = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
