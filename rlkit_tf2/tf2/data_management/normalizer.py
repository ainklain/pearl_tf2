import tensorflow as tf
import numpy as np

from rlkit_tf2.data_management.normalizer import Normalizer, FixedNormalizer

class TF2Normalizer(Normalizer):
    """
    Update with np array, but de/normalize pytorch Tensors.
    """
    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = tf.convert_to_tensor(self.mean, dtype=tf.float32)
        std = tf.convert_to_tensor(self.std, dtype=tf.float32)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = tf.expand_dims(mean, axis=0)
            std = tf.expand_dims(std, axis=0)
        return tf.clip_by_value((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean = tf.convert_to_tensor(self.mean, dtype=tf.float32)
        std = tf.convert_to_tensor(self.std, dtype=tf.float32)
        if v.dim() == 2:
            mean = tf.expand_dims(mean, axis=0)
            std = tf.expand_dims(std, axis=0)
        return mean + v * std


class TorchFixedNormalizer(FixedNormalizer):
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = tf.convert_to_tensor(self.mean, dtype=tf.float32)
        std = tf.convert_to_tensor(self.std, dtype=tf.float32)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = tf.expand_dims(mean, axis=0)
            std = tf.expand_dims(std, axis=0)
        return tf.clip_by_value((v - mean) / std, -clip_range, clip_range)

    def normalize_scale(self, v):
        """
        Only normalize the scale. Do not subtract the mean.
        """
        std = tf.convert_to_tensor(self.std, dtype=tf.float32)
        if v.dim() == 2:
            std = tf.expand_dims(std, axis=0)
        return v / std

    def denormalize(self, v):
        mean = tf.convert_to_tensor(self.mean, dtype=tf.float32)
        std = tf.convert_to_tensor(self.std, dtype=tf.float32)
        if v.dim() == 2:
            mean = tf.expand_dims(mean, axis=0)
            std = tf.expand_dims(std, axis=0)
        return mean + v * std

    def denormalize_scale(self, v):
        """
        Only denormalize the scale. Do not add the mean.
        """
        std = tf.convert_to_tensor(self.std, dtype=tf.float32)
        if v.dim() == 2:
            std = tf.expand_dims(std, axis=0)
        return v * std
