import numpy as np
import keras
import pretty_midi
import librosa

import os
import sys
import mir_eval
import glob

import numpy as np

sys.path.append("..")
sys.path.append("../src")

%load_ext autoreload
%autoreload 
from src import preprocess
from src import constants
from src import datagen
from src.infer import infer_from_processed
from src.metrics import pm_to_pitches_intervals, get_f1_score_notes
from src import helpers

def get_note_evaluation(referances, predictions):
    padded =  np.zeros((predictions.shape[0], 128))
    padded[:, 21:109] = predictions
    padded =  padded.T
    sequence_est = helpers.piano_roll_to_pretty_midi(padded, fs=32, program=0).instruments[0]

    padded =  np.zeros((referances.shape[0], 128))
    padded[:, 21:109] = referances
    padded =  padded.T
    sequence_ref = helpers.piano_roll_to_pretty_midi(padded, fs=32, program=0).instruments[0]
    
    est_intervals,est_pitches = pm_to_pitches_intervals(sequence_est)
    ref_intervals,ref_pitches = pm_to_pitches_intervals(sequence_ref)
    
    note_precision, note_recall, note_f1 = get_f1_score_notes(ref_intervals, ref_pitches, est_intervals, est_pitches)
    return note_precision, note_recall, note_f1

def test(data_dir):
    preprocess.preprocess(data_dir, False)
    model = '../bin/models/final_model.h5'
    test_generator = datagen.DataGenerator(constants.TEST_PROCESSED_DIR, constants.SEQUENCE_SIZE, constants.BATCH_SIZE, constants.CONTEXT_WINDOW_SIZE, 'x_[0-9]+.npy', 'y_[0-9]+.npy')

    note_f1s = []
    note_precisions = []
    note_recalls = []

    for sample in glob.glob(os.path.join(constants.TEST_PROCESSED_DIR, 'x_*')):
        isolated_filename = sample.split('/')[-1]
        
        gt_file= os.path.join(constants.TEST_PROCESSED_DIR,test_generator.corresponding_y(isolated_filename))
        referances = np.load(gt_file, mmap_mode='r')
        predictions = infer_from_processed(model, sample)
        
        note_precision, note_recall, note_f1 = get_note_evaluation(referances, predictions)

        note_f1s.append(note_f1)
        note_precisions.append(note_precision)
        note_recalls.append(note_recall)

    avg_note_f1 = np.mean(note_f1s)
    avg_note_precision = np.mean(note_precisions)
    avg_note_recall = np.mean(note_recalls)

if __name__ == '__main__':
    test(sys.argv[1])