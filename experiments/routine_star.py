

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


def routine(sess, curves, y_values, curves_test, y_test, batch_size, sample_size, num_examples, num_examples_test, reuse_2=False):
    # create GAN
    g = GAN(batch_size, sample_size, sample_size)
    z, y, x = g.create_placeholders()
    Gz = g.generative(z, y, reuse=reuse_2)
    Dx = g.discriminative(x, y, reuse=reuse_2)
    Dg = g.discriminative(Gz, y, reuse=True)
    print('here')
    d_loss, g_loss = g.get_losses(Dx, Dg, Gz)
    trainerD, trainerG = g.get_optimizers(d_loss, g_loss)
    dL = []
    gL = []
    G_losses = []
    D_losses = []
    sess.run(tf.global_variables_initializer())
    g_rounds = 1
    d_rounds = 1
    dLoss = 1
    gLoss = 1
    g_max_loss = 1.
    d_max_loss = 1.
    g_max_run = 10
    for epoch in range(150):
        epoch_start_time = time.time()
        for i in range(num_examples // batch_size):
            z_batch = np.random.normal(0, 1, size=[batch_size, sample_size])
            real_curve_batch = curves[i * batch_size:(i + 1) * batch_size, :]
            y_ = y_values[i * batch_size:(i + 1) * batch_size, :]

            for i in range(d_rounds):
                _, dLoss = sess.run([trainerD, d_loss], feed_dict={
                                    z: z_batch, x: real_curve_batch, y: y_})
                
                if dLoss < d_max_loss:
                    print('yes' + str(dLoss))
                    i = g_max_run
                    break
            for i in range(g_rounds):
                _, gLoss = sess.run([trainerG, g_loss], feed_dict={
                                    z: z_batch, y: y_, x: real_curve_batch})
                
                #print(curves.shape)
                #print(sess.run(Dg, feed_dict={z: curves, y: y_, x: real_curve_batch}))
                if gLoss < g_max_loss:
                    print('yes' + str(gLoss))
                    i = g_max_run
                    break
        d_rounds = 10
        g_rounds = 10
        G_losses.append(float(gLoss))
        D_losses.append(float(dLoss))

    print('ok')
    curves_out = []
    t_out = []
    #all_ts = np.reshape(np.arange(0, 12, .1), [-1,sample_size])
    
    for i in range(num_examples_test// batch_size):
        print(y_test.shape)
        z_batch = np.random.normal(0, 1, size=[batch_size, sample_size])
        # real_curve_batch = curves_test[i * batch_size:(i + 1) * batch_size, :]
        y_ = y_test[i * batch_size:(i + 1) * batch_size, :]
        print(y_.shape)
        curv = sess.run(Gz, feed_dict={z: z_batch, y: y_}) #x:real_curve_batch})
        for c in range(len(curv)):
          curves_out.append(curv[c])
          t_out.append(y_[c])
#discriminator_1 = sess.run(Dg, feed_dict={z: z_batch, y: y_, x: real_curve_batch})
#   discriminator_2 = sess.run(Dx, feed_dict={z: z_batch, y: y_, x: real_curve_batch})
    saver = tf.train.Saver()
    save_path = saver.save(sess=sess, save_path="/tmp/model_sam{}.noise{}.ckpt".format(num_examples, sample_size))
    print(len(curves_out))
    
    sess.close()
    return t_out, curves_out


def main():
    
    #  sample_sizes = [int(sys.argv[1]), int(sys.argv[2])]
    sample_0 = 20
    sample_1 = 200

    values = 80
    
    seed = int(sys.argv[1])
    tot_ = 150

    # seed = 4
    np.random.seed(seed)
    results = {}
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    #create sin_data
    data = json.loads(open('./tests/data_light_curves_2.json').read())
    curve_0 = np.asarray(data['2'])
    curve_0_x = curve_0
    curve_0_y = np.asarray([np.arange(values)]*sample_0)
    # curve_0_x, curve_0_y =
    #sample_curves(curve_0, values) #random sampling curves
    #create saw_data
    curve_1 = np.asarray(data['3'])
    curve_1_x = curve_1
    curve_1_y = np.asarray([np.arange(values)]*sample_1)
    #curve_1_x, curve_1_y =
    #sample_curves(curve_1, values)

    
    print(curve_0_y[:sample_0].shape, curve_0_x[:sample_0].shape)
    #GAN first curve
    curve_0_y_test = np.asarray([np.arange(values)]*(tot_-sample_0))
    t_0_g_out, curves_0_g_out  = routine(
                                         sess, curve_0_x[:sample_0], curve_0_y[:sample_0],curve_0_x[sample_0:tot_-sample_0],curve_0_y_test, 10, values, sample_0, tot_-sample_0, False)
    print('first')

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    #GAN second curve
    curve_1_y_test = np.asarray([np.arange(values)]*(tot_-sample_1))

    t_1_g_out, curves_1_g_out  = routine(
                                         sess, curve_1_x[:sample_1], curve_1_y[:sample_1],curve_1_x[sample_1:tot_-sample_1],curve_1_y_test, 10, values, sample_1, tot_-sample_1, False)

#   CLASSIFIER ON REAL VALUES
    train_data_classifier_curve_0 = np.concatenate((curve_0_x[:sample_0], curve_0_y[:sample_0]), axis=1)
    train_data_classifier_curve_1 = np.concatenate((curve_1_x[:sample_1], curve_1_y[:sample_1]), axis=1)
    train_data_classifier = np.concatenate((train_data_classifier_curve_0, train_data_classifier_curve_1), axis=0)
    
    
    classificatore_real = classifier(values)
    cl_real = classificatore_real.create_clas()
    num_rounds = 600
    
    label_x = [[int(i)] for i in np.zeros(sample_0)]
    label_y = [[int(i)] for i in np.ones(sample_1)]
    labels = np.concatenate((label_x, label_y), axis=0)
    classificatore_real.train(train_data_classifier, labels, cl_real, num_rounds)
    
    
    
    
    #CLASSIFIER ON GAN VALUES
    
    
    train_x_0 = np.concatenate((curve_0_x[:sample_0], curves_0_g_out), axis=0)
    train_y_0 = np.concatenate((curve_0_y[:sample_0], t_0_g_out), axis=0)
    
    train_x_1 = curve_1_x[:sample_1]
    # train_x_1 = np.concatenate((curve_1_x[:sample_1], curves_1_g_out), axis=0)
    train_y_1 = curve_1_y[:sample_1]
    #train_y_1 = np.concatenate((curve_1_y[:sample_1], t_1_g_out), axis=0)
    
    train_data_classifier_curve_0 = np.concatenate((train_x_0[:tot_], train_y_0[:tot_]), axis=1)
    train_data_classifier_curve_1 = np.concatenate((train_x_1[:tot_], train_y_1[:tot_]), axis=1)
    train_data_classifier = np.concatenate((train_data_classifier_curve_0, train_data_classifier_curve_1), axis=0)
    
    
    classificatore = classifier(values)
    cl = classificatore.create_clas()
    num_rounds = 600
    
    label_x = [[int(i)] for i in np.zeros(tot_)]
    label_y = [[int(i)] for i in np.ones(tot_)]
    labels = np.concatenate((label_x, label_y), axis=0)
    classificatore.train(train_data_classifier, labels, cl, num_rounds)
    
    
    
    curve_0_y_ = np.asarray([np.arange(values)]*100)

    test_data_classifier_curve_0 = np.concatenate((curve_0_x[sample_0:sample_0+100], curve_0_y_), axis=1)
    test_data_classifier_curve_1 = np.concatenate((curve_1_x[sample_1:sample_1+100], curve_0_y_), axis=1)
    test_data = np.concatenate((test_data_classifier_curve_0, test_data_classifier_curve_1), axis=0)
    label_x_test = [[int(i)] for i in np.zeros(100)]
    label_y_test = [[int(i)] for i in np.ones(100)]
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
