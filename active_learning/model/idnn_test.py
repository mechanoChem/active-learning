import sys, os
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Concatenate, Reshape
import tensorflow.keras.backend as K
from active_learning.model.transform_layer import Transform
import tensorflow as tf
import numpy as np

class IDNN(tf.keras.Model):
  
    def __init__(self, input_dim, hidden_units, activation='softplus', dropout=None, transforms=None, unique_inputs=False, final_bias=False):
        super().__init__()

        self.transforms = transforms
        self.unique_inputs = unique_inputs

        # Define input layer separately
        self.input_layer = Input(shape=(input_dim,))

        # Define dense layers
        self.dnn_layers = []
        self.dnn_layers.append(Dense(hidden_units[0], activation=activation[0]))
        for i in range(1, len(hidden_units)):
            self.dnn_layers.append(Dense(hidden_units[i], activation=activation[i]))
            if dropout:
                self.dnn_layers.append(Dropout(dropout))
        self.dnn_layers.append(Dense(1, use_bias=final_bias))
        
        # Build the model using the Sequential API for simplicity
        self.model = tf.keras.Sequential([self.input_layer] + self.dnn_layers)

    @tf.function(autograph=False)
    def call(self, inputs):

        def DNN(y, T):
            if self.transforms:
                y = Transform(self.transforms)(y)
            y = tf.keras.layers.concatenate([y, T])
            for layer in self.dnn_layers:
                y = layer(y)
            return y

        if self.unique_inputs:
            x1 = inputs[0]
            x2 = inputs[1]
            x3 = inputs[2]
            x4 = inputs[3]
            y = DNN(x1, x4)

            with tf.GradientTape() as g:
                g.watch(x2)
                y2 = DNN(x2, x4)
            dy = g.gradient(y2, x2)

            with tf.GradientTape() as g2:
                g2.watch(x3)
                with tf.GradientTape() as g1:
                    g1.watch(x3)
                    y3 = DNN(x3, x4)
                dy3 = g1.gradient(y3, x3)
            ddy = g2.batch_jacobian(dy3, x3)

        else:
            x1 = inputs[0]
            x4 = inputs[1]
            with tf.GradientTape() as g2:
                g2.watch(x1)
                with tf.GradientTape() as g1:
                    g1.watch(x1)
                    y = DNN(x1, x4)
                dy = g1.gradient(y, x1)
            ddy = g2.batch_jacobian(dy, x1)

        return [y, dy, ddy]
