from flask import Flask, render_template, request, jsonify, send_file
from flask import Flask, request, jsonify
import random
import json
import re
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import random
import json
import re
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_chatbot():
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

    bot_name = ""

    return model, intents, all_words, tags, bot_name


def chat_with_bot(model, intents, all_words, tags, bot_name, user_input):
    match = re.search(r"my name is (.+)", user_input.lower())
    if match:
        user_name = match.group(1).strip()
        return f"Hi {user_name}! How can I assist you?"

    sentence = tokenize(user_input)
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
                response = random.choice(intent['responses'])
                return f"{response}"
    else:
        return f"I do not understand..."


def chat_with_user(user_input):
    model, intents, all_words, tags, bot_name = initialize_chatbot()
    response = chat_with_bot(model, intents, all_words, tags, bot_name, user_input)
    return response


app = Flask(__name__)


def get_received_text():
    try:
        data = request.form if request.form else request.get_json()
        text = data.get('text')
        return text

    except Exception as e:
        response = {'status': 'error', 'message': str(e)}
        return jsonify(response), 400


def generate_response_from_text(text):
    try:
        user_input = text
        response = chat_with_user(user_input)
        generated_response = response
        return generated_response
    except Exception as e:
        response = {'status': 'error', 'message': str(e)}
        return jsonify(response), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_response', methods=['POST'])
def generate_image():
    try:
        text = get_received_text()
        response_data = generate_response_from_text(text)
        return response_data
    except Exception as e:
        response = {'status': 'error', 'message': str(e)}
        return jsonify(response), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

