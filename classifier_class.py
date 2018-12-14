
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
# %matplotlib inline



"""**CLASSIFICATOR FOR ACCURACY**"""


class classifier():
    def __init__(self, num_values):
        self.num_values = num_values

    def cnn_model_fn(self, features, labels, mode):
        """Model function for CNN."""
        
        input_layer = tf.reshape(features['x'], [-1, 1, self.num_values*2, 1])
        print(input_layer.shape)
        # conv1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=25,
            kernel_size=[1, 5],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1, pool_size=[1, 2], strides=2)

        # conv2
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=50,
            kernel_size=[1, 5],
            padding="same",
            activation=tf.nn.relu)
        print(conv2.shape)
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2, pool_size=[1, 2], strides=2)

        # dense_layer
        pool2_flat = tf.reshape(pool2, [-1, 1 * pool2.shape[2] * 50])
        dense = tf.layers.dense(
            inputs=pool2_flat, units=10 * 25 * self.num_values, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        print(dropout.shape)
        # logits
        logits = tf.layers.dense(inputs=dropout, units=2)
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # loss functions
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

        # training
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # evaluation metric
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                                            labels=labels, predictions=predictions["classes"]), "precision":
tf.metrics.precision(
                     labels=labels, predictions=predictions["classes"]), "recall":tf.metrics.recall(
                                                                                                      labels=labels, predictions=predictions["classes"]),
"auc":tf.metrics.accuracy(
                          labels=labels, predictions=predictions["classes"])
}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def create_clas(self):
        cl = tf.estimator.Estimator(
            model_fn=self.cnn_model_fn)
        return cl

    def train(self, train_data, labels, cl, epochs):
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=700)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': train_data}, y=labels, batch_size=10, num_epochs=None, shuffle=True)
        cl.train(input_fn=train_input_fn, steps=epochs, hooks=[logging_hook])

    def valuate(self, eval_data, labels, cl):
        print(eval_data.shape)
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data}, y=labels, num_epochs=1, shuffle=False)
        eval_results = cl.evaluate(input_fn=eval_input_fn)
        return eval_results
