import numpy as np
import json

import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer

def load_dataset(file):
    '''
    Loading Data .csv file 
    '''
    with open(file, 'r') as f:
        labels = []
        text = []

        lines = f.readlines()

    lines.pop(0)

    for line in lines:
        data = line.split(',', 3)
        if len(data) == 4:
            labels.append(data[1])
            text.append(data[3].rstrip())
    return text[:100000],labels[:100000]

def preprocess_data(X_train, Y_train, max_words):
    '''
    Preprocessing the loaded data
    '''
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)

    X_train = pad_sequences(sequences, maxlen=10)
    Y_train = keras.utils.to_categorical(Y_train, 2)

    # Taking the dictionary of max words
    dictionary = dict(list(dictionary.items())[:max_words])
    # Saving the dictionary
    with open('output/dictionary.json', 'w') as dictionary_file:
        json.dump(dictionary, dictionary_file)

    return X_train, Y_train

