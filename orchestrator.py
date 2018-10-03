import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
import time
import sys
import json
import pickle
from gan_class import GAN
from sample_data_class import sample_data
from classifier_class import classifier
from scipy import signal


def routine(sess, curves, y_values, curves_test, y_test, batch_size, sample_size, num_examples, num_examples_test, reuse_2=False):
    # create GAN
    g = GAN(batch_size, sample_size, sample_size)
    z, y, x = g.create_placeholders()
    Gz = g.generative(z, y, reuse=reuse_2)
    Dx = g.discriminative(x, y, reuse=reuse_2)
    Dg = g.discriminative(Gz, y, reuse=True)
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


    curves_out = []
    t_out = []
    #all_ts = np.reshape(np.arange(0, 12, .1), [-1,sample_size])

    for i in range(num_examples_test// batch_size):
        z_batch = np.random.normal(0, 1, size=[batch_size, sample_size])
        real_curve_batch = curves_test[i * batch_size:(i + 1) * batch_size, :]
        y_ = y_test[i * batch_size:(i + 1) * batch_size, :]
    
        curv = sess.run(Gz, feed_dict={z: z_batch, y: y_, x: real_curve_batch})
        for c in range(len(curv)):
            curves_out.append(curv[c])
            t_out.append(y_[c])
    discriminator_1 = sess.run(Dg, feed_dict={z: z_batch, y: y_, x: real_curve_batch})
    discriminator_2 = sess.run(Dx, feed_dict={z: z_batch, y: y_, x: real_curve_batch})
    saver = tf.train.Saver()
    save_path = saver.save(sess=sess, save_path="/tmp/model_sam{}.noise{}.ckpt".format(num_examples, sample_size))
    print(len(curves_out), len(curves_out[0]))
    
    sess.close()
    return Gz, discriminator_1,discriminator_2,G_losses, D_losses, t_out, curves_out


def main():
    # sample_sizes = [20]
    sample_sizes = [int(sys.argv[1])]
    sample = sample_sizes[0]
    # noise_sizes = [15]
    
    noise_sizes = [int(sys.argv[2])]
    values = noise_sizes[0]
    
    seed = int(sys.argv[3])
    # seed = 4
    np.random.seed(seed)
    results = {}
    
    len_t = sample
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    #create sin_data
    sin_data = sample_data(
                           'sin', n_samples=sample*4, no_drop=values, reuse=False, mul_range=[1, 2])
    sin_data.create_points()
                           
    #create saw_data
    saw_data = sample_data('saw', n_samples=sample*4, mul_range=[
                                                                                        1, 2], no_drop=values, pred=1)
    saw_data.create_points()
                           
                           #GAN sin
    Gz_sin, d_g_sin, d_x_sin, g_loss_sin, d_loss_sin, y_out_sin, curves_g_sin  = routine(
                                                                                                                sess, sin_data.y_values[:len_t], sin_data.x_values[:len_t],sin_data.y_values[len_t:], sin_data.x_values[len_t:], 10, values, len_t, len_t*3, False)
                           
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
                           #GAN saw
    Gz_saw, d_g_saw, d_x_saw, g_loss_saw, d_loss_saw, y_out_saw, curves_g_saw  = routine(
                                                                                                                sess, saw_data.y_values[:len_t], saw_data.x_values[:len_t],saw_data.y_values[len_t:], saw_data.x_values[len_t:], 10, values, len_t,len_t*3, False)
                           
    #CLASSIFIER GAN
    #train_data_classifier
    train_data_classifier_sin = np.concatenate((curves_g_sin[:len_t*2], y_out_sin[:len_t*2]), axis=1)
    train_data_classifier_saw = np.concatenate((curves_g_saw[:len_t*2], y_out_saw[:len_t*2]), axis=1)
    train_data_classifier = np.concatenate((train_data_classifier_sin, train_data_classifier_saw), axis=0)
   
    #test_data_classifier
    test_data_classifier_sin = np.concatenate((curves_g_sin[len_t*2:], y_out_sin[len_t*2:]), axis=1)
    test_data_classifier_saw = np.concatenate((curves_g_saw[len_t*2:], y_out_saw[len_t*2:]), axis=1)
    test_data_classifier = np.concatenate((test_data_classifier_sin, test_data_classifier_saw), axis=0)
   
    #classifier train on gan
    classificatore = classifier(values)
    cl = classificatore.create_clas()
    num_rounds = 400
    label_x = [[int(i)] for i in np.zeros(len_t*2)]
    label_y = [[int(i)] for i in np.ones(len_t*2)]
    labels = np.concatenate((label_x, label_y), axis=0)
   
    label_x_test = [[int(i)] for i in np.zeros(int(len_t))]
    label_y_test = [[int(i)] for i in np.ones(int(len_t))]
    labels_test = np.concatenate((label_x_test, label_y_test), axis=0)
    acc_class = 0
    print(train_data_classifier.shape, labels.shape)
   
    while acc_class < 0.1:
        classificatore.train(train_data_classifier, labels, cl, num_rounds)
        print(test_data_classifier.shape, labels_test.shape)
        acc_class = classificatore.valuate(test_data_classifier, labels_test, cl)['accuracy']
        print(acc_class)
        num_rounds += 100
    results['accuracy_classifier'] = float(acc_class)


    #CLASSIFIER REAL
    #train_data_classifier
    sin_data = sample_data(
                       'sin', n_samples=len_t*3, no_drop=values, reuse=False, mul_range=[1, 2])
    sin_data.create_points()
    
    #create saw_data
    saw_data = sample_data('saw', n_samples=len_t*3, mul_range=[
                                                                1, 2], no_drop=values, pred=1)
    saw_data.create_points()
    train_data_classifier_sin = np.concatenate((sin_data.y_values[:len_t*2], sin_data.x_values[:len_t*2]), axis=1)
    train_data_classifier_saw = np.concatenate((saw_data.y_values[:len_t*2],saw_data.x_values[:len_t*2]), axis=1)
    train_data_classifier = np.concatenate((train_data_classifier_sin, train_data_classifier_saw), axis=0)
                                                                
                                                                #test_data_classifier
    test_data_classifier_sin = np.concatenate((sin_data.y_values[len_t*2:], sin_data.x_values[len_t*2:]), axis=1)
    test_data_classifier_saw = np.concatenate((saw_data.y_values[len_t*2:], saw_data.x_values[len_t*2:]), axis=1)
    test_data_classifier = np.concatenate((test_data_classifier_sin, test_data_classifier_saw), axis=0)
                                                                
    #classifier train on gan
    classificatore_real = classifier(values)
    cl_real = classificatore_real.create_clas()
    num_rounds = 400
    label_x = [[int(i)] for i in np.zeros(len_t*2)]
    label_y = [[int(i)] for i in np.ones(len_t*2)]
    labels = np.concatenate((label_x, label_y), axis=0)
                                                                
    label_x_test = [[int(i)] for i in np.zeros(int(len_t))]
    label_y_test = [[int(i)] for i in np.ones(int(len_t))]
    labels_test = np.concatenate((label_x_test, label_y_test), axis=0)
    acc_class = 0
    print(train_data_classifier.shape, labels.shape)
                                                                
    while acc_class < 0.1:
        classificatore_real.train(train_data_classifier, labels, cl_real, num_rounds)
        print(test_data_classifier.shape, labels_test.shape)
        acc_class = classificatore_real.valuate(test_data_classifier, labels_test, cl_real)['accuracy']
        print(acc_class)
        num_rounds += 100
    results['accuracy_classifier_real'] = float(acc_class)

    #create new curves sin
    #create sin_data
    sin_data = sample_data(
                       'sin', n_samples=sample, no_drop=values, reuse=False, mul_range=[1, 2])
    sin_data.create_points()
    
    #create saw_data
    saw_data = sample_data('saw', n_samples=sample, mul_range=[
                                                               1, 2], no_drop=values, pred=1)
    saw_data.create_points()
                                                               
    #train_data_sin_
    test_data_sin = np.concatenate((sin_data.y_values, sin_data.x_values), axis=1)
                                                               
    #train_data_saw_
    test_data_saw = np.concatenate((saw_data.y_values, saw_data.x_values), axis=1)
                                                               
    test_data = np.concatenate((test_data_sin, test_data_saw), axis=0)
                                                               
                                                               
    #evaluate classifier on real curve
    acc_class_real = classificatore.valuate(test_data, labels_test, cl)['accuracy']
    acc_class_real_real = classificatore_real.valuate(test_data, labels_test, cl_real)['accuracy']
                                                               
    results['accuracy_real'] = float(acc_class_real)
    results['accuracy_real_real'] = float(acc_class_real_real)
                                                               
    results['loss_sin'] = [g_loss_sin, d_loss_sin]
    results['loss_saw'] = [g_loss_saw, d_loss_saw]
                                                               
    out_file_ac = 'results.s{}.sam{}.noise{}.json'.format(seed, sample_sizes, noise_sizes)
    open(out_file_ac, 'w').write(json.dumps(results))
    out_file_curves = 'curves.s{}.sam{}.noise{}.json'.format(seed, sample_sizes[0], noise_sizes[0])
    open(out_file_curves, 'wb').write(pickle.dumps([[curves_g_sin,y_out_sin], [curves_g_saw,y_out_saw]]))
                                                               
                                                               
    return results

if __name__ == "__main__":
    
    main()
