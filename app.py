import json
import random
from tkinter import *

import torch
from charlie import NeuralNet
import logging
from charlie import bag_of_words, tokenize

logging.basicConfig(filename='sample.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('charlie/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = 'data.pth'

data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    # specification of a confidence level
    if prob.item() > 0.65:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return ("Sorry! Can't help you with this one. Try this from my Friend Google https://www.google.com.")



class ChatApplication:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Talk to your GYMSHARK Virtual Assistant")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=500, height=550, bg="#17202A")

        # head label

        head_label = Label(self.window, bg="#17202A", fg="#EAECEE", text=("Charlie the Chatbot"),
                           font="Helvetica 16 bold", pady=10)
        head_label.place(relwidth=1)

        # tiny divider

        line = Label(self.window, width=480, bg="#1E70EB")
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget

        self.text_widget = Text(self.window, width=20,
                                height=2, bg="#17202A", fg="#EAECEE", font="Helvetica 12", padx=20, pady=20, wrap=WORD)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # scroll bar

        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.999)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = Label(self.window, bg="#17202A", height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg="white", font="Helvetica 14", insertbackground="white")
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # send button

        send_button = Button(bottom_label, text="Send", font="Helvetica 16 bold", fg="white", width=20, bg="#1E70EB",
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "YOU")

    def _insert_message(self, msg, sender):
        bot_name = 'Charlie'
        if not msg:
            return

        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)


if __name__ == '__main__':
    app = ChatApplication()
    app.run()
