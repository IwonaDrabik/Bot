#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Dependencies

# In[34]:


#Please install if necessary

# pip install numpy
# pip install nltk
# pip install json
# pip install torch
# pip install tkinter
# pip install matplotlib.pyplot
# pip install seaborn
# pip install pandas
# pip install collections


# In[1]:


import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

#import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tkinter
from tkinter import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import collections


# # 2. Data Loading

# In[2]:


#Please adjust the path if necessary (when the input location differs from the present .ipynb/Python code location)

with open("C:\\Users\\56idr\\Desktop\\Charlie\\intents.json") as file:
	intents = json.load(file)


# # 3. Tokenization and Dataset Creation

# In[3]:


def tokenize(sentence):
    
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    
    """
    stemming = find the root form of the word
    examples:
    words = ["black", "blacken", "blackened"]
    words = [stem(w) for w in words]
    -> ["black", "black", "black"]
    
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hi", "my", "name", "is","charlie"]
    words = ["hi", "hello", "my", "you", "bye", "thank", "cool"]
    bag   = [  1 ,    0 ,      1 ,   0 ,    0 ,    0 ,      0]
    
    """
    # stem each word
    
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # initialize bag with 0 for each word
    
    bag = np.zeros(len(words), dtype=np.float32)
    
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag


# In[4]:


a = "is it raining?"
print(a)


# In[5]:


a = tokenize(a)
print(a)


# In[6]:


words = ["draw", "draws", "drawing"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)


# In[7]:


all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))


# In[8]:


ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]


# ## 3.1. Visualizing Words Frequency
# (just out of curiosity)

# In[9]:


from collections import Counter
Counter(all_words)


# In[10]:


a = np.array(all_words)
x = collections.Counter(a)
from collections import Counter
x= x.most_common(15)
df1 = pd.DataFrame(x, columns=["word", "frequency"])
df1


# In[11]:


plt.figure(figsize=(15,8))
ax = sns.barplot(x="frequency", y="word", data=df1, palette="mako")
plt.xlabel("Number of Units", size=15)
plt.ylabel("Lemma", size=15)
plt.title("Most Frequent Words", size=15)


# ## 3.2. Showing the number of words, patterns and tags

# In[12]:


# Assign data of lists 
data = {'Item': ['tags','patterns', 'words'], 'Count': [len(tags), len(xy),len(all_words)]}  
  
# Create DataFrame  
df2 = pd.DataFrame(data)  
  
# Print the output.  
df2


# In[13]:


import seaborn as sns
plt.figure(figsize=(15,8))
ax = sns.barplot(x="Item", y="Count", data=df2,
                 palette="Blues_r",order=df2.sort_values('Count',ascending = False).Item)
plt.xlabel("Item", size=15)
plt.ylabel("Number of Verbal Units", size=15)
plt.title("Tokenized Input Units", size=18)


# In[14]:


# stem and lower each word

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)


# # 4. Model Building and Data Training 

# In[15]:


# Create training data

x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


# In[16]:


# Creating hyper-parameters 

num_epochs = 2000
batch_size = 10
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 10
output_size = len(tags)
print(input_size, output_size)


# In[17]:


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# In[18]:


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)


# In[19]:


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


# In[20]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)


# In[21]:


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses=[]
accuracies=[]

# Train the model
for epoch in range(num_epochs):
    total_train = 0
    correct_train = 0
    
    # enumerate mini batches
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Run the forward pass
        outputs = model(words)
        
        # Calculate loss
        
        loss = criterion(outputs, labels)
        losses.append(loss)
        
        # Backprop and perform Adam optimization
        # gradients clearance
        optimizer.zero_grad()
        
       
        # Credit assignment
        loss.backward()
        
        # Update model weights
        optimizer.step()
 

        # Track the accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()
        train_accuracy = correct_train / total_train
        accuracies.append(train_accuracy)
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy}')

print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')


# In[22]:


input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]


# In[23]:


model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


# ## 4.1.  Accuracy over iterations 

# In[24]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(15, 10), dpi=80)
plt.plot(accuracies)

#For better visual impression, have limited the number of iteration in the chart below to 2000.
#For inspecting the entire result you can comment out the line below:

plt.xlim([0, 2000])

#Labels
plt.xlabel(r'Iterations', fontsize=20)
plt.ylabel(r'Accuracy', fontsize=20)


# # Stage 5: Defining a Chatbot

# In[25]:


# Defining a Chatbot

bot_name = "CHARLIE"

def get_response(msg):
#     print("Let's chat!")
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
        for  intent in intents['intents']:
            if tag == intent["tag"]:
            
                return random.choice(intent['responses'])

    return("Sorry! Can't help you with this one. Try this from my Friend Google https://www.google.com.")


# # Chatbot GUI Application with Tkinter 

# In[26]:


# GUI: translating the Chatbot into a visual form
# Please go to the dialogue window that will automatically open upon code execution

BG_GRAY = "#1E70EB"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 12"
FONT_BOLD = "Helvetica 16 bold"
   
class ChatApplication:

    def __init__(self):
        
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        
        self.window.mainloop()

        
        
    def _setup_main_window(self):
    
        self.window.title("Talk to your GYMSHARK Virtual Assistant")
        self.window.resizable(width=False, height = False)
        self.window.configure(width=500, height=550,bg=BG_COLOR)
        
        
        # head label
        
        head_label = Label(self.window, bg=BG_COLOR,fg=TEXT_COLOR, text=("Charlie the Chatbot"),
                           font = FONT_BOLD, pady=10)
        head_label.place(relwidth=1)
        
        # tiny divider
        
        line = Label(self.window, width=480, bg = BG_GRAY)
        line.place(relwidth = 1, rely = 0.07, relheight = 0.012)

        
        # text widget
        
        self.text_widget = Text(self.window, width = 20,
                                height=2,bg=BG_COLOR,fg=TEXT_COLOR, font=FONT, padx = 20, pady = 20,wrap=WORD)
        self.text_widget.place(relheight=0.745, relwidth=1,rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        
        # scroll bar
        
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.999)
        scrollbar.configure(command = self.text_widget.yview)
        
        # bottom label
        bottom_label = Label(self.window, bg=BG_COLOR, height = 80)
        bottom_label.place(relwidth=1,rely = 0.825)
        
        # message entry box
        self.msg_entry = Entry(bottom_label, bg = "#2C3E50",fg="white", font = "Helvetica 14",insertbackground="white")
        self.msg_entry.place(relwidth=0.74, relheight = 0.06, rely = 0.008, relx = 0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        
        # send button    
        
        send_button = Button(bottom_label, text = "Send", font = FONT_BOLD, fg="white", width=20,bg=BG_GRAY,
                            command = lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008,relheight=0.06,relwidth=0.22)
        
    def _on_enter_pressed(self, event):
        
        msg = self.msg_entry.get()
        self._insert_message(msg, "YOU")
        
    def _insert_message(self, msg, sender):
               
        if not msg:
             return
      
        #msg_intro="My name is Charlie."
 
        self.msg_entry.delete(0,END)
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

