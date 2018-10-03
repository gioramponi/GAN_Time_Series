

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


def create_slices(curves, size=32):
    new_curves = []
    new_y = []
    for j in range(len(curves)):
        for i in range(int(len(curves[j])/size)):
            new_curves.append(curves[j][i*size:(i+1)*size])
            new_y.append(np.arange(len(curves[j]))[i*size:(i+1)*size])
    return np.asarray(new_curves), np.asarray(new_y)



def main():
    
    #  sample_sizes = [int(sys.argv[1]), int(sys.argv[2])]
    sample_0 = 27
    sample_1 = 93
    
    values = 96
    
    size = 32
    num = int(values/size)
    
    seed = 4
    np.random.seed(seed)
    results = {}
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    #create sin_data
    data = json.loads(open('ecg_data.json').read())
    curve_0 = np.asarray(data['-1'])
    curve_0_x = curve_0
    curve_0_y = np.asarray([np.arange(values)]*sample_0)
    # curve_0_x, curve_0_y =
    #sample_curves(curve_0, values) #random sampling curves
    #create saw_data
    curve_1 = np.asarray(data['1'])
    curve_1_x = curve_1
    curve_1_y = np.asarray([np.arange(values)]*sample_1)
    #curve_1_x, curve_1_y =
    #sample_curves(curve_1, values)
    
    
    slice_x_0, slice_y_0 = create_slices(curve_0_x, size)
    
    slice_x_1, slice_y_1 = create_slices(curve_1_x, size)
    print('sl', slice_x_0.shape, slice_x_1.shape)
    
    
    train_x_0 = slice_x_0[:num*sample_0]
    train_y_0 = slice_y_0[:num*sample_0]
    
    train_x_1 = slice_x_1[:num*sample_1]
    train_y_1 = slice_y_1[:num*sample_1]
    print('k', train_x_0.shape, train_x_1.shape)
    
    
    #CLASSIFIER ON GAN VALUES
    
    
    
    #train_x_1 = curve_1_x[:sample_1]
    train_data_classifier_curve_0 = np.concatenate((train_x_0, train_y_0), axis=1)
    train_data_classifier_curve_1 = np.concatenate((train_x_1, train_y_1), axis=1)
    
    train_data_classifier = np.concatenate((train_data_classifier_curve_0, train_data_classifier_curve_1), axis=0)
    
    
    classificatore = classifier(size)
    cl = classificatore.create_clas()
    num_rounds = 900
    
    label_x = [[int(i)] for i in np.zeros(train_data_classifier_curve_0.shape[0])]
    label_y = [[int(i)] for i in np.ones(train_data_classifier_curve_1.shape[0])]
    labels = np.concatenate((label_x, label_y), axis=0)
    classificatore.train(train_data_classifier, labels, cl, num_rounds)
    
    
    curve_0_y_ = np.asarray([np.arange(size)]*(slice_x_0.shape[0]-train_data_classifier_curve_0.shape[0]))
    
    curve_1_y_ = np.asarray([np.arange(size)]*(slice_x_1.shape[0]-train_data_classifier_curve_1.shape[0]))
    
    print('u',slice_x_1[sample_1*num:sample_1*num+40*num].shape,curve_1_y_.shape )
    #    print(curve_1_y_.shape, slice_x_1[sample_1*num:sample_0*num+40*num].shape)
    test_data_classifier_curve_0 = np.concatenate((slice_x_0[sample_0*num:sample_0*num+40*num], curve_0_y_), axis=1)
    
    test_data_classifier_curve_1 = np.concatenate((slice_x_1[sample_1*num:sample_1*num+40*num], curve_1_y_), axis=1)
    test_data = np.concatenate((test_data_classifier_curve_0, test_data_classifier_curve_1), axis=0)
    label_x_test = [[int(i)] for i in np.zeros(40*num)]
    label_y_test = [[int(i)] for i in np.ones(40*num)]
    labels_test = np.concatenate((label_x_test, label_y_test), axis=0)
    
    
    #evaluate classifier on real curve
    acc_class_real = list(classificatore.epredict(test_data, labels_test, cl))
    precision = 0
    count = 0
    for k in range(len(acc_class_real)):
        if int(acc_class_real[k]['classes']) == 1:
            count+=1
        if (k+1)%3 == 0:
            if count>=2 and labels[k] == 1:
                precision +=1
            if count<2 and labels[k] == 0:
                precision += 1
            count = 0
    print(precision)



    #  acc_class_real = classificatore.valuate(test_data, labels_test, cl)['accuracy']
    #    acc_class_real_real = classificatore_real.valuate(test_data, labels_test, cl_real)['accuracy']

    results['accuracy_real'] = precision/(len(acc_class_real)/3)
    #results['accuracy_real_real'] = float(acc_class_real_real)
    
    #  results['loss_sin'] = [g_loss_sin, d_loss_sin]
    #   results['loss_saw'] = [g_loss_saw, d_loss_saw]
    
    out_file_ac = 'results.s{}.sam{}.noise{}.json'.format(
                                                          seed, sample_0, values)
    open(out_file_ac, 'w').write(json.dumps(results))
                                                          
                                                          
                                                          # out_file_curves = 'curves.s{}.sam{}.noise{}.json'.format(seed, sample_0, values)
                                                          # open(out_file_curves, 'wb').write(pickle.dumps([[curves_0_g_out,t_0_g_out], [curves_1_g_out,t_1_g_out]]))
                                                          
                                                          
    return results

if __name__ == "__main__":
    
    main()
