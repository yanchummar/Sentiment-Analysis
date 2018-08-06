import numpy as np
import json

from preprocess import load_dataset, preprocess_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Dropout, Flatten, AveragePooling1D
from keras.layers.embeddings import Embedding

# Getting Preprocessed Data

# word dictionary length
max_words = 10000

# loading data from twitter sentiment analysis dataset
print('Loading Data...')
X, Y = load_dataset('data/dataset.csv')
# preprocessing
print('Preprocessing Data...')
X_train, Y_train = preprocess_data(X, Y, max_words)

# Model Architecture

embedding_size = 10
input_dim = max_words+1
input_length = X_train.shape[1]

model = Sequential()
model.add(Embedding(input_dim=input_dim, output_dim=embedding_size, input_length=input_length, name='embedding_input_layer'))
model.add(Conv1D(32, 3, padding='same'))
model.add(Conv1D(32, 3, padding='same'))
model.add(Conv1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dense(2,activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training 
print('Training...')
model.fit(X_train, Y_train, epochs=10, verbose=1, validation_split=0.1, batch_size=32)

# Saving Model and Weights
model_json = model.to_json()
with open('output/model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('output/model.h5')