import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt

# Load intents file
with open('intents.json') as file:
    data = json.load(file)

# Preprocess the data
lemmatizer = WordNetLemmatizer()
intents = data['intents']
words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']

for intent in intents:
    for pattern in intent['patterns']:
        # Tokenize words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add to documents
        documents.append((w, intent['tag']))
        # Add to classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)
training = np.array(training)

# Split data into features and labels
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Build and train the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Train the model and store history for plotting
history = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Plotting accuracy over epochs
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
# Save the model
model.save('chatbot_model.h5')


# Save words, classes, and training data
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open('training_data.pkl', 'wb'))

# Load the model and data
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('training_data.pkl', 'rb'))['words']
classes = pickle.load(open('training_data.pkl', 'rb'))['classes']

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")

    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    
    # Check if the detected intent is in the predefined intents
    if any(i['tag'] == tag for i in list_of_intents):
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        result = "I'm sorry, I didn't understand that."
    
    return result



def chatbot_response(msg):
    ints = predict_class(msg, model)
    
    # Check if the detected intent has sufficient confidence
    if ints[0]['probability'] < '0.25':
        return "I'm sorry, I didn't understand that."
    
    res = get_response(ints, intents)
    return res


# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    response = chatbot_response(user_input)
    print("ChatBot:", response)

