import librosa as libr
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import IPython.display as ipd

import glob
import constants

import os
import sys

midi_path_test = 'data/test/MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--1.midi'
wav_path_test = 'data/test/MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--1.wav'

midi_path = 'data/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--1.midi'
wav_path = 'data/train/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--1.wav'


def get_corresponding_y(x_f):
    y_file = x_f.split('.')[0] + '.midi'
    return y_file


def preprocess_one_file(x_file, y_file, ctr, istrain):

    print('x file is: {}'.format(x_file))
    audio, sample_rate = libr.load(x_file, sr=16000)
    ipd.Audio(audio, rate=sample_rate)
    print('Audio shape is {}'.format(audio.shape))

    proc_input = libr.cqt(audio, window='hamming', sr=sample_rate, hop_length=512, fmin=constants.FMIN, n_bins=constants.N_BINS, bins_per_octave=constants.BINS_P_OCTAVE)
    proc_input = np.abs(proc_input)
    print('after CQT shape is {} '.format(proc_input.shape))

    #frame amount: 4751. piece length: 152 seconds. frame rate: 4751/152 = 31.25 frame/sec
    print("printing first three:")
    print(proc_input[:3])

    proc_input= proc_input.T
    print('transposing {}'.format(proc_input.shape))


    mean = np.mean(proc_input, axis=0)
    std = np.std(proc_input, axis=0)
    print('mean is {}'.format(mean.shape))
    print('std is {}'.format(std.shape))

    # proc_input_norm = (proc_input - mean)/std
    proc_input_norm = proc_input
    print('noamrlized is {}'.format(proc_input_norm.shape))

    x_out = proc_input_norm
    print('outputting x shape {}'.format(x_out.shape))

    times = libr.frames_to_time(np.arange(proc_input_norm.shape[0]), sr=sample_rate, hop_length=512)

    print('retrieving times with shape {}'.format(times.shape))


    print('y file is: {}'.format(y_file))
    pm = pretty_midi.PrettyMIDI(y_file)

    piano_roll = pm.get_piano_roll(times=times)[constants.MIN_MIDI:constants.MAX_MIDI+1].T
    print('got piano roll with shape{}'.format(piano_roll.shape))
    piano_roll[piano_roll > 0] = 1
    y_out = piano_roll

    to_pad =  int(np.ceil(x_out.shape[0]/constants.SEQUENCE_SIZE) * constants.SEQUENCE_SIZE - x_out.shape[0])
    x_out = np.append(x_out, np.zeros((to_pad, x_out.shape[1])), axis = 0)
    y_out = np.append(y_out, np.zeros((to_pad, y_out.shape[1])), axis = 0)
    print('leffover padding is {}'.format(to_pad))
    print(np.zeros((to_pad, x_out.shape[1])).shape)

    print('finally getting X  with shape {}'.format(x_out.shape))
    print('finally getting Y with shape {}'.format(y_out.shape))

    print('finally padding for context window of size {}'.format(constants.CONTEXT_WINDOW_SIZE))
    pad_size = constants.CONTEXT_WINDOW_SIZE // 2
    x_out = np.append(x_out, np.zeros((pad_size, x_out.shape[1])), axis = 0)
    x_out = np.insert(x_out, 0, np.zeros((pad_size,  x_out.shape[1])), axis = 0)

    save_dir = None
    if istrain:
        save_dir = constants.TRAIN_PROCESSED_DIR
    else:
        save_dir = constants.TEST_PROCESSED_DIR
    x_file_path = os.path.join(save_dir, 'x_{}'.format(ctr))
    y_file_path = os.path.join(save_dir, 'y_{}'.format(ctr))
    np.save(x_file_path, x_out)
    np.save(y_file_path, y_out)

    print('saved {}'.format(x_file_path))
    print('saved {}'.format(y_file_path))



def preprocess_files(dirname, x_files, istrain=True):
    ctr = 0
    for x_f in x_files:
        y_f = get_corresponding_y( x_f)
        preprocess_one_file(x_f, y_f, ctr, istrain)
        ctr+=1

def preprocess(data_path):
    test_path = os.path.join(data_path, 'test')
    train_path = os.path.join(data_path, 'train')

    test_files = glob.glob(os.path.join(test_path, '*.wav'))
    train_files = glob.glob(os.path.join(train_path, '*.wav'))

    preprocess_files(train_path, train_files, istrain=True)
    preprocess_files(test_path, test_files, istrain=False)

if __name__ == '__main__':
    preprocess(sys.argv[1])