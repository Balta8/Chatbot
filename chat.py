import random
import json
import re
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"
memory = {"interesting_input": []}  # Initialize memory dictionary with a list for interesting input

print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence="do you know content creation?"
    sentence = input("You: ")
    if sentence == "quit":
        break
    match = re.search(r"my name is (.+)", sentence.lower())
    if match:
        user_name = match.group(1).strip()
        print(f"{bot_name}: Hi {user_name}! How can I assist you?")
        continue  # Skip the rest of the loop and prompt for the next user input


    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                if intent.get("interesting"):
                    memory["interesting_input"].append(sentence)
    else:
        print(f"{bot_name}: I do not understand...")

# Print or use the collected interesting input
print(f"{bot_name}: I found the following interesting input: {memory['interesting_input']}")
    