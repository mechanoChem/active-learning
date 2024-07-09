import tensorflow as tf
import sys, os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
import tensorflow.keras.backend as K
directory = os.path.abspath('../../../active-learning/')
sys.path.insert(0,directory)
from active_learning.model.transform_layer import Transform

class IDNN(tf.keras.Model):
    def __init__(self, input_dim, hidden_units, activation='softplus', dropout=None, transforms=None, unique_inputs=False, final_bias=False):
        super().__init__()
        
        self.transforms = transforms
        self.unique_inputs = unique_inputs

        # Define the input layers
        if self.unique_inputs:
            self.input_layer_1 = Input(shape=(input_dim,))
            self.input_layer_2 = Input(shape=(input_dim,))
            self.input_layer_3 = Input(shape=(input_dim,))
            self.input_layer_4 = Input(shape=(input_dim,))
        else:
            self.input_layer_1 = Input(shape=(input_dim,))
            self.input_layer_4 = Input(shape=(input_dim,))

        # Define dense layers
        self.dnn_layers = []
        self.dnn_layers.append(Dense(hidden_units[0], activation=activation[0]))
        for i in range(1, len(hidden_units)):
            self.dnn_layers.append(Dense(hidden_units[i], activation=activation[i]))
            if dropout:
                self.dnn_layers.append(Dropout(dropout))
        self.dnn_layers.append(Dense(1, use_bias=final_bias))

    def call(self, inputs):
        def DNN(y, T):
            if self.transforms:
                y = Transform(self.transforms)(y)
            y = Concatenate()([y, T])
            for layer in self.dnn_layers:
                y = layer(y)
            return y

        if self.unique_inputs:
            x1, x2, x3, x4 = inputs
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
            x1, x4 = inputs
            with tf.GradientTape() as g2:
                g2.watch(x1)
                with tf.GradientTape() as g1:
                    g1.watch(x1)
                    y = DNN(x1, x4)
                dy = g1.gradient(y, x1)
            ddy = g2.batch_jacobian(dy, x1)

        return [y, dy, ddy]

# Example usage:
input_dim = 4  # Replace with your input dimension
hidden_units = [64, 64, 64]  # Replace with your hidden layers configuration
activation = ['relu', 'relu', 'relu']  # Replace with your activations
dropout = 0.2  # Replace with your dropout rate if needed
transforms = None  # Replace with your transform functions if needed
unique_inputs = True  # Change based on your requirement
final_bias = True  # Change based on your requirement

model = IDNN(input_dim, hidden_units, activation, dropout, transforms, unique_inputs, final_bias)

# Example of compiling and training the model
model.compile(optimizer='adam', loss='mse')

# Replace with your actual training data
x_train = [tf.random.normal((100, input_dim)),tf.random.normal((100, input_dim)),tf.random.normal((100, input_dim)), tf.random.normal((100, 1))]
y_train = [tf.random.normal((100, 1)), tf.random.normal((100, input_dim)), tf.random.normal((100, input_dim, input_dim))]

model.fit(x_train, y_train, epochs=10, batch_size=32,verbose=0)
