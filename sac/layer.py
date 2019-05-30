# https://www.tensorflow.org/alpha/guide/keras/custom_layers_and_models

from myrl.base import Policy

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Flatten, Dense



class Linear(layers.Layer):
    def __init__(self,
                 units=32,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros'):
        super().__init__()
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer=self.bias_initializer,
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {'units': self.units}








