from tensorflow.python.keras.utils import Sequence
import glob
import os
import re
import numpy as np

class DataGenerator(Sequence):

    def __init__(self, directory, sequence_len, batch_size, window_size, x_regex, y_regex):
        self.directory = directory
        self.batch_size =batch_size
        self.sequence_len = sequence_len
        self.window_size =window_size
        self.x_regex = x_regex
        self.y_regex = y_regex
        self.x_files = []
        self.y_files = []
        self.files = os.listdir(directory)
        self.data_size = 0
        for f in self.files:
            if re.search(self.x_regex, f):
                self.x_files.append(f)
            elif re.search(self.y_regex, f):
                self.y_files.append(f)
                self.data_size += np.load(os.path.join(self.directory,f), mmap_mode='r').shape[0]
        print(self.data_size)
        print('now listing inputs')
        print(self.x_files)
        print('now listing outputs')
        print(self.y_files)

        self.curfile = 0
        self.curFrameInd = 0
            

    def __len__(self):
        return self.data_size // self.sequence_len

    def __getitem__(self):
        for f in self.x_files:
            cur_x = np.load(os.path.join(self.directory,f), mmap_mode='r')
            cur_y = np.load(os.path.join(self.directory,self.corresponding_y(f)), mmap_mode='r')
            size = cur_x.shape[0]
            print(size)
            hf_win = self.window_size//2
            print(hf_win)
            for i in range(hf_win, size, self.sequence_len):
                if( i + self.sequence_len > size):
                    print(str(i) + 'breaking')
                    # return
                    yield None
                cur_x_sequence = cur_x[i - hf_win : i + self.sequence_len + hf_win]
                cur_y_sequence =  cur_y[i - hf_win: (i - hf_win) + self.sequence_len]
                x_seq = []
                y_seq = []
                for j in range(self.sequence_len):
                    frame_window = cur_x_sequence[j : j + self.window_size]
                    frame_window =  np.expand_dims(frame_window, axis = 2)
                    note = cur_y_sequence[j]
                    x_seq.append(frame_window)
                    y_seq.append(note)
                x_seq=np.expand_dims(np.array(x_seq), axis = 0)
                y_seq=np.expand_dims(np.array(y_seq), axis = 0)
                print(x_seq.shape)
                print(y_seq.shape)
                yield x_seq, y_seq
    

    def __getitemtest__(self):
        for f in self.x_files:
            cur_x = np.load(os.path.join(self.directory,f), mmap_mode='r')
            size = cur_x.shape[0]
            print(size)
            hf_win = self.window_size // 2
            print(hf_win)
            for i in range(hf_win, size, self.sequence_len):
                if( i + self.sequence_len > size):
                    print('one up')
                cur_x_sequence = cur_x[i - hf_win : i + self.sequence_len + hf_win]
                print(i + self.sequence_len + hf_win)
                x_seq = []
                for j in range(self.sequence_len):
                    frame_window = cur_x_sequence[j : j + self.window_size]
                    frame_window =  np.expand_dims(frame_window, axis = 2)
                    x_seq.append(frame_window)
                    if(i + self.sequence_len + hf_win == 4806):
                        print(frame_window.shape)
                        print(len(x_seq))
                x_seq=np.expand_dims(np.array(x_seq), axis = 0)
                print(x_seq.shape)
                yield x_seq

                

    def corresponding_y(self, x_filename):
        y_filename = 'y' + x_filename[1:]
        if y_filename in self.y_files:
            print(y_filename)
            return y_filename
        else:
            raise Exception

    