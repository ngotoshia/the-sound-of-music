import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model

import pretty_midi

import helpers 
import constants
from preprocess import preprocess_one_file_wav
from datagen import DataGenerator

def infer(model_name, filename, output_name):
    model = load_model(model_name)
    print('preprocessing file {}'.format(filename))
    processed_file = preprocess_one_file_wav(filename, 0)
    print('infering from model "{}"'.format(model_name))
    predictions = model.predict_generator(generator=DataGenerator.__genfileforinference__(processed_file, constants.CONTEXT_WINDOW_SIZE, constants.SEQUENCE_SIZE),
                                                steps=DataGenerator.__inferencelen__(processed_file, constants.SEQUENCE_SIZE))

    print('postprocessing')
    predictions = np.reshape(predictions, (predictions.shape[0]*predictions.shape[1], predictions.shape[2]))
    predictions = np.rint(predictions)

    padded =  np.zeros((predictions.shape[0], 128))
    padded[:, 21:109] = predictions
    padded =  padded.T

    midi_file_pm = helpers.piano_roll_to_pretty_midi(padded, fs=32, program=0)
    print('outputting to "{}"'.format(output_name))
    midi_file_pm.write(output_name)
    return output_name

if __name__ == '__main__':
    infer(sys.argv[1], sys.argv[2], sys.argv[3])