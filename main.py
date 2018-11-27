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
    for epoch in range(250):
        epoch_start_time = time.time()
        for i in range(num_examples // batch_size):
            z_batch = np.random.normal(0, 1, size=[batch_size, sample_size])
            real_curve_batch = curves[i * batch_size:(i + 1) * batch_size, :]
            y_ = y_values[i * batch_size:(i + 1) * batch_size, :]

            for i in range(d_rounds):
                _, dLoss = sess.run([trainerD, d_loss], feed_dict={
                                    z: z_batch, x: real_curve_batch, y: y_})
                
                if dLoss < d_max_loss:
                    i = g_max_run
                    break
            for i in range(g_rounds):
                _, gLoss = sess.run([trainerG, g_loss], feed_dict={
                                    z: z_batch, y: y_, x: real_curve_batch})
                if gLoss < g_max_loss:
                    i = g_max_run
                    break
        d_rounds = 10
        g_rounds = 10
        G_losses.append(float(gLoss))
        D_losses.append(float(dLoss))

    curves_out = []
    t_out = []
    
    for i in range(num_examples_test// batch_size):
        print(y_test.shape)
        z_batch = np.random.normal(0, 1, size=[batch_size, sample_size])
        y_ = y_test[i * batch_size:(i + 1) * batch_size, :]
        print(y_.shape)
        curv = sess.run(Gz, feed_dict={z: z_batch, y: y_}) #x:real_curve_batch})
        for c in range(len(curv)):
          curves_out.append(curv[c])
          t_out.append(y_[c])

    saver = tf.train.Saver()
    save_path = saver.save(sess=sess, save_path="/tmp/model_sam{}.noise{}.ckpt".format(num_examples, sample_size))
    print(len(curves_out))
    
    sess.close()
    return t_out, curves_out


def main():
    
    sample_sizes = [int(sys.argv[1])]
    sample_0 = sample_sizes[0]

    values = int(sys.argv[2])
    
    tot_ = 100

    np.random.seed(1)
    results = {}
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    file_in = str(sys.argv[3])

    file_times = str(sys.argv[4])
    file_out = str(sys.argv[5])
    
    
    curve =json.loads(open(file_in).read())
    
    curve_x = np.asarray(curve['values'])
    curve_y = np.asarray(curve['time'])
    times = np.asarray(pickle.loads(open(file_times,'rb').read()))

    t_0_g_out, curves_0_g_out  = routine(
                                         sess, curve_x, curve_y,curve_x,times, 10, values, sample_0, times.shape[0], False)
    
    
    
    open(file_out, 'wb').write(pickle.dumps([curves_0_g_out,t_0_g_out]))
    
    
    return results

if __name__ == "__main__":
    
    main()
