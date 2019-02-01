# Sentiment Analysis
Sentiment Analysis using Convolutional Neural Networks with Keras

### About
Sentiment analysis using CNNs built using Keras with a **validation accuracy around 79%**

### Requirements
> - Keras
> - Tensorflow
> - Numpy

### Training Dataset
The model is trained on a portion of Twitter's Sentiment Analysis Dataset, you can download it [here](http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip).

The dataset as a csv file is stored in data folder and is preprocessed and used for training.

The model gets a validation accuracy of around **79%** by training on 15,78,627 entries of the Twitter dataset over 5 epochs.

### Setup

Run the ```classify.py``` in your terminal to test the Sentiment Analysis model on any text. 

The code for preproccessing is in the ```preprocess.py``` and the model along with code to train the model is in ```model.py```

### Model Architecture

A simple sequential architecture consisting of few Convolutional layers and the Fully Connected Layer.
