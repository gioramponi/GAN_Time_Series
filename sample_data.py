import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
import time
import sys
import json
import pickle
from scipy import signal


class sample_data():
    def __init__(self, curve_type, n_samples=10000, t_start=0.,t_inputs=12, no_drop=45, max_offset=100, mul_range=[1, 2], reuse=False, pred=0, noise=False):
        self.n_samples = n_samples  # number of samples
        self.t_inputs = t_inputs  # initial interval
        self.no_drop = no_drop  # number of random time points
        self.max_offset = max_offset
        self.mul_range = mul_range
        self. reuse = reuse  # if reuse the same time points #for error?
        self.pred = pred  # prediction
        self.noise = noise  # if noise
        self.x_values = []
        self.y_values = []
        self.t_start = t_start
        self.curve_type = curve_type
    
    def create_points(self):
        for i in range(self.n_samples):
            noise = 0
            mul = self.mul_range[0] + np.random.random() * \
                (self.mul_range[1] - self.mul_range[0])
            offset = np.random.random() * self.max_offset
            if self.noise:
                noise = np.random.random() * 0.3
            if self.curve_type == 'saw':
                if not self.reuse:
                    x_vals_samp = np.sort(np.random.uniform(t_start,self.t_inputs,self.no_drop))
                else:
                    x_vals_samp = self.x_values[i]
                self.y_values.append(signal.sawtooth(
                                                     2 * np.pi * 5 * x_vals_samp / 12) + noise)
                self.x_values.append(x_vals_samp)
            if self.curve_type == 'sin':
                if not self.reuse:
                    x_vals_samp = np.sort(np.random.uniform(0.0,self.t_inputs,self.no_drop))
                else:
                    x_vals_samp = x_vals[i]     #
                self.y_values.append(
                                     np.sin(x_vals_samp * mul) / 2 + .5 + noise)
                self.x_values.append(x_vals_samp)
        self.y_values = np.asarray(self.y_values)
        self.x_values = np.asarray(self.x_values)
