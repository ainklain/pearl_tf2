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


class MLPBlock(layers.Layer):
    def __init__(self,
                 hidden_sizes,
                 output_size,
                 hidden_activation='relu',
                 output_activation='linear'):
        super().__init__()
        self.hidden_layers = list()
        for i, h in enumerate(hidden_sizes):
            self.hidden_layers.append(Dense(h, activation=hidden_activation))
        self.output_layers = Dense(output_size, activation=output_activation)

    def call(self, inputs):
        x = self.hidden_layers[0](inputs)
        if len(self.hidden_layers) > 1:
            for i in range(1, len(self.hidden_layers)):
                x = self.hidden_layers[i](x)

        return self.output_layers(x)


class FlattenMLP(layers.Layer):
    def __init__(self, *args, **kwargs):
        self.mlp = MLPBlock(*args, **kwargs)
        self.flatten = Flatten()

    def call(self, inputs):
        flatten_input = self.flatten(inputs)
        return self.mlp(flatten_input)


class MLPPolicy(Model, Policy):
    def __init__(self,
                 hidden_sizes,
                 output_size,
                 *args,
                 obs_normalizer=None,
                 **kwargs):
        super().__init__()
        self.obs_normalizer = obs_normalizer

        self.mlp = MLPBlock(hidden_sizes, output_size, *args, **kwargs)

    def call(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return self.mlp(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :]

    def get_actions(self, obs):
        return self.call(obs).numpy()


class TanhMLPPolicy(MLPPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation='tanh')

class MLPEncoder(FlattenMLP):
    def __init__(self):
        super().__init__()

    def reset(self, num_tasks=1):
        pass

# class RecurrentEncoder(FlattenMLP):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)









