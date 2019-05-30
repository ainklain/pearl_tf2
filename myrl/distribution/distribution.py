# from myrl.distribution.base import Distribution
import tensorflow_probability.python.distributions as tfd
from tensorflow_probability.python.distributions import Distribution, Normal

import numpy as np
import tensorflow as tf


class DiagonalGaussian(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl(self, old_dist_info, new_dist_info):
        old_means = old_dist_info['mean']
        old_log_stds = old_dist_info['log_std']

        new_means = new_dist_info['mean']
        new_log_stds = new_dist_info['log_std']

        old_std = np.exp(old_log_stds)
        new_std = np.exp(new_log_stds)
        # means: N * A
        # stds: N * A
        # formula: {(mu1 - mu2)^2 + sig1^2 - sig2^2} / (2*sig2^2) + ln(sig2 / sig1)
        numerator = np.square(old_means - new_means) + np.square(old_std) - np.square(new_std)
        denominator = 2 * np.square(new_std) + 1e-8

        return np.sum(numerator / denominator + new_log_stds - old_log_stds, axis=-1)

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_means = old_dist_info_vars['mean']
        old_log_stds = old_dist_info_vars['log_std']
        new_means = new_dist_info_vars['mean']
        new_log_stds = new_dist_info_vars['log_std']

        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)

        numerator = tf.square(old_means - new_means) + tf.square(old_std) - tf.square(new_std)
        denominator = 2 * tf.square(new_std) + 1e-8
        return tf.reduce_sum(numerator / denominator + new_log_stds - old_log_stds, reduction_indices=-1)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return tf.exp(logli_new - logli_old + 1e-6)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        means = dist_info_vars['mean']
        log_stds = dist_info_vars['log_std']
        zs = (x_var - means) / tf.exp(log_stds)
        return - tf.reduce_sum(log_stds, axis=-1) \
               - 0.5 * tf.reduce_sum(tf.square(zs), axis=-1) \
               - 0.5 * self.dim * np.log(2 * np.pi)

    def sample(self, dist_info):
        means = dist_info['mean']
        log_stds = dist_info['log_std']
        rnd = np.random.normal(size=means.shape)
        return rnd * tf.exp(log_stds) + means

    def log_likelihood(self, xs, dist_info):
        means = dist_info['mean']
        log_stds = dist_info['log_std']
        zs = (xs - means) / tf.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - 0.5 * np.sum(np.square(zs), axis=-1) - 0.5 * self.dim * np.log(2 * np.pi)

    def entropy(self, dist_info):
        log_stds = dist_info['log_std']
        return tf.reduce_sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    @property
    def dist_info_specs(self):
        return [('mean', (self.dim, )), ('log_std', (self.dim, ))]




class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample(n)
        if return_pre_tanh_value:
            return tf.math.tanh(z), z
        else:
            return tf.math.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """

        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = tf.math.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - tf.math.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.

        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample()

        if return_pretanh_value:
            return tf.math.tanh(z), z
        else:
            return tf.math.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                tf.zeros_like(self.normal_mean),
                tf.ones_like(self.normal_std)
            ).sample()
        )

        if return_pretanh_value:
            return tf.math.tanh(z), z
        else:
            return tf.math.tanh(z)


