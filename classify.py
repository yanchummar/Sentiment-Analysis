import json
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

def classify():
    # we're still going to use a Tokenizer here, but we don't need to fit it
    tokenizer = Tokenizer(num_words=10000)
    # for human-friendly printing
    labels = ['negative', 'positive']

    # read in our saved dictionary
    with open('output/dictionary.json', 'r') as dictionary_file:
        dictionary = json.load(dictionary_file)

    # this utility makes sure that all the words in your input
    # are registered in the dictionary
    # before trying to turn them into a matrix.
    def convert_text_to_index_array(text):
        words = kpt.text_to_word_sequence(text)
        wordIndices = []
        for word in words:
            if word in dictionary:
                wordIndices.append(dictionary[word])
            else:
                print("'%s' not in training corpus; ignoring." %(word))
        return wordIndices

    # read in your saved model structure
    json_file = open('output/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # and create a model from that
    model_pred = model_from_json(loaded_model_json)
    # and weight your nodes with your saved values
    model_pred.load_weights('output/model.h5')

    # okay here's the interactive part
    input_text = input('Text to classify:')

    # format your input for the neural net
    text_indices = convert_text_to_index_array(input_text)
    predict_text = pad_sequences([text_indices], maxlen=10)
    
    # predict which bucket your input belongs in
    pred = model_pred.predict(predict_text)
    # and print it for the humons
    print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))

classify()