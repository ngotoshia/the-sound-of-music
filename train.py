import os 
import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Conv2D, MaxPooling2D, Dropout, LSTM, Bidirectional, Flatten 
from sklearn.metrics import auc, f1_score, precision_score, recall_score, roc_curve

import constants
import datagen

def get_model():
    model = Sequential()
    model.add(TimeDistributed(
        Conv2D(64, (3,3), activation='relu'), \
            input_shape=(constants.SEQUENCE_SIZE, constants.CONTEXT_WINDOW_SIZE,constants.FREQUENCIES_SIZE,1)
        )
    )
    model.add(Dropout(0.5))
    model.add(TimeDistributed(MaxPooling2D((1,3), strides=(1,1))))
            
    model.add(TimeDistributed(Conv2D(128, (1,3), activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(MaxPooling2D((1,3), strides=(1,1))))

    model.add(TimeDistributed(Flatten()))

    model.add(TimeDistributed(Dense(200)))
    model.add(Dropout(0.5))

    # model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5), merge_mode ='ave'))

    model.add(TimeDistributed(Dense(constants.INFERED_NOTE_N, activation='sigmoid')))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
          
def train(model_name):
    keras.backend.clear_session()
    model = get_model()
    inp = model.input     # input placeholder
    outputs = [(layer.input,layer.output) for layer in model.layers]          # all layer outputs
    print ('\n'.join(map(str,outputs)))
    train_generator = datagen.DataGenerator(constants.TRAIN_PROCESSED_DIR, constants.SEQUENCE_SIZE, constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, 'x_[0-9]+.npy', 'y_[0-9]+.npy')
    model.fit_generator(generator=train_generator.__getitem__(),
                    use_multiprocessing=False,
                     epochs=constants.EPOCHS, steps_per_epoch=train_generator.__len__())
    model.save(model_name)

if __name__ == '__main__':
    train(sys.argv[1])