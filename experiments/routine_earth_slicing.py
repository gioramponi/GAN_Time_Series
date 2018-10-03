

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
import time
import sys
import json
import pickle
import random
from scipy import signal
from sample_data_class import sample_data
from gan_class import GAN
from classifier_class import classifier
# %matplotlib inline


def sample_curves(curves,noise):
    num = np.arange(1024)
    y_values = []
    x_values = []
    for curve in curves:
        y_val = sorted(random.sample(list(num), noise))
        x_ = []
        for v in range(noise):
            x_.append(curve[y_val[v]])
        x_values.append(x_)
        y_values.append(y_val)
    return np.asarray(x_values), np.asarray(y_values)


def create_slices(curves, size=8):
    new_curves = []
    new_y = []
    for j in range(len(curves)):
        for i in range(int(len(curves)/size)-1):
            print((i+1)*size)
            new_curves.append(curves[j][i*size:(i+1)*size])
            new_y.append(np.arange(len(curves[j]))[i*size:(i+1)*size])
    return np.asarray(new_curves), np.asarray(new_y)


def main():
    
    #  sample_sizes = [int(sys.argv[1]), int(sys.argv[2])]
    sample_0 = 100
    sample_1 = 200

    values = 24
    
    seed = int(sys.argv[1])
    tot_ = 300
    size = 8
    num = int(values/size)
    # seed = 4
    np.random.seed(seed)
    results = {}
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    #create sin_data
    data = json.loads(open('PowerDemand.json').read())
    curve_0 = np.asarray(data['1'])
    curve_0_x = curve_0
    curve_0_y = np.asarray([np.arange(values)]*sample_0)
    # curve_0_x, curve_0_y =
    #sample_curves(curve_0, values) #random sampling curves
    #create saw_data
    curve_1 = np.asarray(data['2'])
    curve_1_x = curve_1
    curve_1_y = np.asarray([np.arange(values)]*sample_1)
    #curve_1_x, curve_1_y =
    #sample_curves(curve_1, values)

    
    slice_x_0, slice_y_0 = create_slices(curve_0_x, size)
    
    slice_x_1, slice_y_1 = create_slices(curve_1_x, size)
    
    train_x_0 = slice_x_0[:num*sample_0]
    train_y_0 = slice_y_0[:num*sample_0]
    
    train_x_1 = slice_x_1[:num*sample_1]
    train_y_1 = slice_y_1[:num*sample_1]
    
    
    
    #CLASSIFIER ON GAN VALUES
    
    
    train_x_0 = np.concatenate((curve_0_x[:sample_0], curves_0_g_out), axis=0)
    train_y_0 = np.concatenate((curve_0_y[:sample_0], t_0_g_out), axis=0)
    
    #train_x_1 = curve_1_x[:sample_1]
    print(np.asarray(curve_1_x).shape, np.asarray(curves_1_g_out).shape)
    train_x_1 = np.concatenate((curve_1_x[:sample_1], curves_1_g_out), axis=0)
    #train_y_1 = curve_1_y[:sample_1]
    train_y_1 = np.concatenate((curve_1_y[:sample_1], t_1_g_out), axis=0)
    
    train_data_classifier_curve_0 = np.concatenate((train_x_0[:tot_], train_y_0[:tot_]), axis=1)
    train_data_classifier_curve_1 = np.concatenate((train_x_1[:tot_], train_y_1[:tot_]), axis=1)
    
    train_data_classifier = np.concatenate((train_data_classifier_curve_0, train_data_classifier_curve_1), axis=0)
    
    
    classificatore = classifier(values)
    cl = classificatore.create_clas()
    num_rounds = 900
    
    label_x = [[int(i)] for i in np.zeros(train_data_classifier_curve_0.shape[0])]
    label_y = [[int(i)] for i in np.ones(train_data_classifier_curve_1.shape[0])]
    labels = np.concatenate((label_x, label_y), axis=0)
    print(labels.shape,train_data_classifier.shape)
    classificatore.train(train_data_classifier, labels, cl, num_rounds)
    
    
    
    curve_0_y_ = np.asarray([np.arange(values)]*200)

    test_data_classifier_curve_0 = np.concatenate((curve_0_x[sample_0:sample_0+200], curve_0_y_), axis=1)
    test_data_classifier_curve_1 = np.concatenate((curve_1_x[sample_1:sample_1+200], curve_0_y_), axis=1)
    test_data = np.concatenate((test_data_classifier_curve_0, test_data_classifier_curve_1), axis=0)
    label_x_test = [[int(i)] for i in np.zeros(200)]
    label_y_test = [[int(i)] for i in np.ones(200)]
    labels_test = np.concatenate((label_x_test, label_y_test), axis=0)
    

    #evaluate classifier on real curve
    acc_class_real = classificatore.valuate(test_data, labels_test, cl)['accuracy']
    acc_class_real_real = classificatore_real.valuate(test_data, labels_test, cl_real)['accuracy']

    results['accuracy_real'] = float(acc_class_real)
    results['accuracy_real_real'] = float(acc_class_real_real)

#  results['loss_sin'] = [g_loss_sin, d_loss_sin]
#   results['loss_saw'] = [g_loss_saw, d_loss_saw]
    
    out_file_ac = 'results.s{}.sam{}.noise{}.json'.format(
        seed, sample_0, values)
    open(out_file_ac, 'w').write(json.dumps(results))    

    
    out_file_curves = 'curves.s{}.sam{}.noise{}.json'.format(seed, sample_0, values)
    open(out_file_curves, 'wb').write(pickle.dumps([[curves_0_g_out,t_0_g_out], [curves_1_g_out,t_1_g_out]]))
    
    
    return results

if __name__ == "__main__":
    
    main()
