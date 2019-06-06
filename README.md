# NN-Sound-Classification-Hackbay2019
Neuronal Network for Classification of hail sound patterns - implemented on Hackbay 2019 as Team "Claimbox" for HUK Coburg Challenge.
mainly based on: https://hackernoon.com/ai-which-classifies-sounds-code-python-6a07a2043810
This script can also work for many other sound recognition projects, depending on your input data.

## Train a Neuronal Network
see hackbayTrainNN.ipynb

First you have to set up you training dataset. I used Kaggles Urban Sound Classification dataset (https://www.kaggle.com/pavansanagapati/urban-sound-classification) as input data for the noHail label and I used YouTube hail sound videos, which I cut in 4 second .wav files (see my repo Audio Splitter) as input data for the Hail label.

Per file, 40 features are extracted using the librosa library and the mfcc feature extraction.

The extractes features are converted into one-hot encoding datasets and are then used to train a Sequential Neuronal Network with 3 hidden layers. 
The final accuracy of the train/test split is over 96%.

The model is exported to a .h5 file

Tip: Use Google Colab or Amazons Deep Learning Studio for training the model. This way you don't have to worry about the necessary hardware and tensorflow installation.

## Use NN for Classification
see hackbayShow

The trained model gets imported and librosas feature extraction function is defined. 
An audio file gets imported, the features get extraced and the model predicts whether the audio file contains hail or not.

## Dependencies
- pandas, numpy
- librosa
- sklearn
- Keras
- Tensorflow