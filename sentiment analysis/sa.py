import os
import pandas as pd 
import tensorflow as tf
import numpy as np

#df = pd.read_csv(os.path.join('jigsaw-toxic-comment-classification-challenge','train.csv', 'train.csv'))
df= pd.read_csv(r"C:\Users\ABI PRIYANKA\OneDrive\Desktop\Projects\sentiment analysis\jigsaw-toxic-comment-classification-challenge\train.csv\train.csv")
df.head()

from tensorflow.keras.layers import TextVectorization
X= df['comment_text']
y=df[df.columns[2:]].values
y
max_features= 200000
vectorizer=TextVectorization(max_tokens=max_features,output_sequence_length=2000,output_mode='int')
vectorizer.adapt(X.values)
with tf.device('/gpu:0'):
    vectorized_text = vectorizer(X.values)
vectorized_text

#data pipeline

dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) 

batch_x,batch_y= dataset.as_numpy_iterator().next()
batch_x
batch_y

train=dataset.take(int(len(dataset)*.7))
valid=dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test=dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Bidirectional,Dense, Embedding

model = Sequential()
# Create the embedding layer 
model.add(Embedding(max_features+1, 32))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer 
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer='Adam')
model.summary()
history = model.fit(train, epochs=1, validation_data=valid)
# Save the model
model.save("toxic_comment_classification_model")


#plotting it after running for multiple epochs
from matplotlib import pyplot as plt
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()



# Load the model
loaded_model = tf.keras.models.load_model("toxic_comment_classification_model")

# Test the loaded model
input_text = ["You freaking suck! I am going to hit you."]

# Convert input text to strings
input_text = tf.constant(input_text, dtype=tf.string)

# Vectorize the input text
input_vectorized_text = vectorizer(input_text)

# Reshape the input data to have 3 dimensions
input_vectorized_text = tf.expand_dims(input_vectorized_text, axis=0)

# Predict
res = loaded_model.predict(input_vectorized_text)

# Convert probabilities to classes
predicted_classes = (res > 0.5).astype(int)
print("Predicted classes:", predicted_classes)

#model evaluation

from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator(): 
    # Unpack the batch 
    X_true, y_true = batch
    # Make a prediction 
    yhat = model.predict(X_true)
    
    # Flatten the predictions
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)

print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
import gradio as gr
model.save('toxicity.h5')
model = tf.keras.models.load_model('toxicity.h5')
input_str = vectorizer('hey i freaken hate you!')
res = model.predict(np.expand_dims(input_str,0))
res
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text
interface = gr.Interface(fn=score_comment, inputs="textbox", outputs="text", title="Comment Scorer")

interface.launch(share=True)