
import sys, os

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Concatenate, Reshape
import tensorflow.keras.backend as K
from transform_layer import Transform
import tensorflow as tf

import numpy as np

class IDNN(Model):
  """
  Integrable deep neural network (idnn)
  Keras model class.

  :param input_dim: Size of input vector
  :type input_dim: int

  :param hidden_units: List containing the number of neurons in each hidden layer
  :type hidden_units: [int]

  :param dropout: Dropout parameter applied after each hidden layer (default is None)
  :type dropout: float

  :param transforms: List of functions to transform the input vector, applied before the first hidden layer (default is None)
  :type transforms: [function]

  :param unique_inputs: if True, requires separate input vectors for the function, its gradient, and its Hessian; if False, assumes the same input vector will be used for function and all derivatives
  :type unique_inputs: bool

  :param final_bias: if True, a bias is applied to the output layer (this cannot be used if only derivative data is used in training); if False, no bias is applied to the output layer (default is False)
  :type final_bias: bool

  The idnn can be trained with first derivative (gradient) data, second derivative (Hessian) data, and/or data from the function itself. (If only derivative data is used then the ``final_bias`` parameter must be ``False``.) The training data for the function and its derivatives can be given at the same input values (in which case, ``unique_inputs`` should be ``False``), or at different input values, e.g. providing the function values at :math:`x \in \{0,1,2,3\}` and the derivative values at :math:`x \in \{0.5,1.5,2.5,3.5\}` (requiring ``unique_inputs`` to be ``True``). Even when ``unique_inputs`` is ``True``, however, the same number of data points must be given for the derivatives and function, even though the input values themselves are different. So, for example, if one had first derivative values at :math:`x \in \{0,1,2,3\}` and second derivative values only at :math:`x \in \{0.5,1.5,2.5\}`, then some of the second derivative data would need to be repeated to that the number of data points are equal, e.g. :math:`x \in \{0.5,1.5,2.5,2.5\}`. Currently, the IDNN structure assumes the function output is a scalar, the gradient is a vector, and the Hessian is a matrix.

  The following is an example where values for the function and the first derivative are used for training, but they are known at different input values. Note that the loss and loss_weights are defined only for the given data (function data and first derivatative data), but fictitious data has to be given for the second derivative or an error will be thrown:

  .. code-block:: python 

     idnn = IDNN(1,
            [20,20],
            unique_inputs=True,
            final_bias=True)

     idnn.compile(loss=['mse','mse',None],
             loss_weights=[0.01,1,None],
             optimizer=keras.optimizers.RMSprop(lr=0.01))

     idnn.fit([c_train0,c_train,0*c_train],
              [g_train0,mu_train,0*mu_train],
              epochs=50000,
              batch_size=20)

  """

  def __init__(self, input_dim,hidden_units,activation='softplus',dropout=None,transforms=None,unique_inputs=False,final_bias=False):
    super().__init__()
    
    self.transforms = transforms
    self.unique_inputs = unique_inputs
  
    # Define dense layers
    self.dnn_layers = []
    self.dnn_layers.append(Dense(hidden_units[0], activation=activation, input_dim=input_dim))
    for i in range(1,len(hidden_units)):
      self.dnn_layers.append(Dense(hidden_units[i], activation=activation))
      if dropout:
        self.dnn_layers.append(Dropout(dropout))
    self.dnn_layers.append(Dense(1,use_bias=final_bias))
        
  @tf.function(autograph=False)
  def call(self, inputs):

    def DNN(y, T):
      if self.transforms:
        y = Transform(self.transforms)(y)
      y =tf.keras.layers.concatenate([y, T])
      for layer in self.dnn_layers:
        y = layer(y)
      return y

    if self.unique_inputs:
      x1 = inputs[0]
      x2 = inputs[1]
      x3 = inputs[2]
      T1 = inputs[3]
      T2 = inputs[4]
      T3 = inputs[5]      
      y = DNN(x1, T1)
      
      with tf.GradientTape() as g:
        g.watch(x2)
        y2 = DNN(x2,T2)
      dy = g.gradient(y2,x2)
      
      with tf.GradientTape() as g2:
        g2.watch(x3)
        with tf.GradientTape() as g1:
          g1.watch(x3)
          y3 = DNN(x3, T3)
        dy3 = g1.gradient(y3,x3)
      ddy = g2.batch_jacobian(dy3,x3)
      
    else:
      x1 = inputs[0]
      T = inputs[1]
      with tf.GradientTape() as g2:
        g2.watch(x1)
        with tf.GradientTape() as g1:
          g1.watch(x1)
          y = DNN(x1, T)
        dy = g1.gradient(y,x1)
      ddy = g2.batch_jacobian(dy,x1)

    return [y,dy,ddy]  