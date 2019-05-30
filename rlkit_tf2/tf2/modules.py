"""
Contain some self-contained modules.
"""
import tensorflow as tf
from tensorflow.keras import layers, losses



# class HuberLoss(layers.Layer):
#     def __init__(self, delta=1):
#         super().__init__()
#         self.huber_loss_delta1 = losses.Huber()
#         self.delta = delta
#
#     def call(self, x, x_hat):
#         loss = self.huber_loss_delta1(x / self.delta, x_hat / self.delta)
#         return loss * self.delta * self.delta
#
#
# class LayerNorm(layers.Layer):
#     """
#     Simple 1D LayerNorm.
#     """
#
#     def __init__(self, features, center=True, scale=False, eps=1e-6):
#         super().__init__()
#         self.center = center
#         self.scale = scale
#         self.eps = eps
#         if self.scale:
#             self.scale_param = tf.ones_like(features)
#         else:
#             self.scale_param = None
#         if self.center:
#             self.center_param = tf.zeros_like(features)
#         else:
#             self.center_param = None
#
#     def call(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         output = (x - mean) / (std + self.eps)
#         if self.scale:
#             output = output * self.scale_param
#         if self.center:
#             output = output + self.center_param
#         return output