import tensorflow as tf
import numpy as np

class GAN():
    
    def __init__(self, batch_size, sample_size, noise_size, keep_prob=0.8):
        self.batch_size = batch_size
        self.out = sample_size
        self.keep_prob = keep_prob
        self.len_samples = sample_size
        self.noise_size = noise_size
    
    def create_placeholders(self):
        z_placeholder = tf.placeholder(dtype=tf.float32, shape=(
                                                                None, self.len_samples))  # placeholder per input_noise
        y_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.len_samples))
        x_placeholder = tf.placeholder(tf.float32, shape=(None, self.len_samples))
        return z_placeholder, y_placeholder, x_placeholder

def generative(self, z, y, reuse=False, is_train=True):
    with tf.variable_scope('generator') as scope:
        if(reuse):
            tf.get_variable_scope().reuse_variables()
            # make conv2_d_traspose
            # make weights
            s = self.out
            s2, s5, s10 = int(s / 2), int(s / 5), int(s / 10)
            z = tf.concat(values=[z, y], axis=1)
            z = tf.reshape(z, [self.batch_size, 1, 1, int(z.get_shape()[1])])
            print('z',z.shape)
            w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
            b_init = tf.constant_initializer(0.0)
            l1 = tf.layers.conv2d_transpose(
                                            inputs=z, filters=s10, kernel_size=1, kernel_initializer=w_init, bias_initializer=b_init)
            l1 = tf.contrib.layers.batch_norm(inputs=l1, center=True, scale=True, is_training=is_train, scope="g_bn1")
            l1 = tf.nn.relu(l1)
                                            
            l2 = tf.layers.conv2d_transpose(inputs=l1, filters=s5, kernel_size=1, kernel_initializer=w_init, bias_initializer=b_init)
            l2 = tf.contrib.layers.batch_norm(inputs=l2, center=True, scale=True, is_training=is_train, scope="g_bn2")
            l2 = tf.nn.relu(l2)
                                            
            l3 = tf.layers.conv2d_transpose(inputs=l2, filters=s2, kernel_size=1, kernel_initializer=w_init, bias_initializer=b_init)
            l3 = tf.contrib.layers.batch_norm(inputs=l3, center=True, scale=True, is_training=is_train, scope="g_bn3")
            l3 = tf.nn.relu(l3)
            l4 = tf.layers.conv2d_transpose(inputs=l3, filters=s, kernel_size=1, kernel_initializer=w_init, bias_initializer=b_init)
                                            #l4 = tf.contrib.layers.fully_connected(l4)
                                            
            return tf.reshape(l4, [self.batch_size, self.out])

def lrelu(self, x, th=0.2):
    return tf.maximum(th * x, x)
    def discriminative(self, x, y, reuse=False, is_train=True):
        keep_prob = self.keep_prob
        with tf.variable_scope('discriminator') as scope:
            if (reuse):
                tf.get_variable_scope().reuse_variables()
            x = tf.concat([x,y],axis=1)
            x = tf.reshape(x,[-1, 2, y.shape[1], 1])
            
            num_values = y.shape[1]
            conv1 = tf.layers.conv2d(
                                     inputs=x,
                                     filters=25,
                                     kernel_size=[1, 5],
                                     padding="same",
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=2)
                                     
                                     # conv2
            conv2 = tf.layers.conv2d(inputs=conv1,filters=50,kernel_size=[1, 5],padding="same",activation=tf.nn.relu)
            print(conv2.shape)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)
                                             
                                             # dense_layer
            pool2_flat = tf.reshape(pool2, [-1, 1 * pool2.shape[2] * 50])
            dense = tf.layers.dense(inputs=pool2_flat, units=10 * 25 * num_values, activation=tf.nn.relu)
            dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)
            print(dropout.shape)
            logits = dropout
                                                     # logits
            logits = tf.layers.dense(inputs=dropout, units=2)
                                                     
            return logits

def get_losses(self, Dx, Dg, Gz):
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(                      logits=Dx, labels=tf.ones_like(Dx)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(                                                                                             logits=Dg, labels=tf.zeros_like(Dg)))
    d_loss = d_loss_fake + d_loss_real
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))
    return d_loss, g_loss

def get_optimizers(self, d_loss, g_loss):
    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if 'discriminator' in var.name]
    print(d_vars)
    g_vars = [var for var in tvars if 'generator' in var.name]
    adam = tf.train.AdamOptimizer()
    trainerD = adam.minimize(d_loss, var_list=d_vars)
    # gen = tf.train.GradientDescentOptimizer(learning_rate=0.8)
    gen = tf.train.AdamOptimizer()
    trainerG = gen.minimize(g_loss, var_list=g_vars)
    return trainerD, trainerG

