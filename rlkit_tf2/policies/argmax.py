"""
Torch argmax policy
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from rlkit.policies.base import Policy


class ArgmaxDiscretePolicy(Model, Policy):
    def __init__(self, qf):
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        q_values = tf.squeeze(self.qf(obs), 0)
        q_values_np = q_values.numpy()
        return q_values_np.argmax(), {}
