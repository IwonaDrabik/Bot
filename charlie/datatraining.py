import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from charlie.model import NeuralNet
from charlie.nlp import bag_of_words, tokenize, stem

import logging

logging.basicConfig(filename='sample.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s')

with open('intents.json', 'r') as file:
    intents = json.load(file)

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

# stemming and applying lower case
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

logging.info(f'patterns {len(xy)}')
logging.info(f'tags: {len(tags)}')
logging.info(f'{len(all_words)}, unique stemmed words: {all_words}')

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

num_epochs = 2000
batch_size = 10
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 10
output_size = len(tags)

logging.info(f'{input_size}, {output_size}')

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

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []
accuracies = []

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

    if (epoch + 1) % 100 == 0:
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy}')

logging.info(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "../../data.pth"
torch.save(data, FILE)
#
logging.info(f'training complete. file saved to {FILE}')
