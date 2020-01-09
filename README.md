# GAN_Time_Series

The model is a Conditional Generative Adversarial Network for time series with not regular time intervals.

The model is created to generate a new time series given a training set of them.



## Why generating data?

The main idea is to use this model to augment the unbalanced dataset of time series, in order to increase the precision of a classifier.


## HOW TO USE THE MODEL

- Requirements:
- python 3
- tensorflow, numpy

- Download the repository

- python3 main.py N M file_in file_times file_out
- N = training set size
- M = time series length
- file_in = file input path
- file_times = file with time stamps of the new time series
- file_out = file output path



If you are interested in this work: https://arxiv.org/abs/1811.08295
