import json
import numpy as np
import os
import random
import glob

class NumpyFileShuffle:
    def get_random_t_data(self, length):
        place = np.random.randint(0,self.tokens_t.shape[0]-length-1)
        return self.tokens_t[place:place+length], self.tokens_t[place+1:place+length+1]
        
    def get_v_data(self):
        return self.tokens_v

    def load_t(self, filename):
        self.tokens_t = np.load(filename)
        
    def load_v(self, filename):
        self.tokens_v = np.load(filename)

    def __init__(self, timesteps, filename_t, filename_v, vocab_size):
        if not os.path.exists(filename_t):
            print(f"Numpy file {filename_t} not found. Aborting.")
            exit(0)
        if not os.path.exists(filename_v):
            print(f"Numpy file {filename_v} not found. Aborting.")
            exit(0)
        self.load_t(filename_t)
        self.load_v(filename_v)
        self.timesteps = timesteps
        self.vocab_size = vocab_size
