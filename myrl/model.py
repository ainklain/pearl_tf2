
from environment import PortfolioEnv

import numpy as np
import tensorflow as tf
import pickle
import pandas as pd


class Argument:
    def __init__(self):
        self.dim_hidden_a = [64, 32, 16]
        self.dim_hidden_c = [32]
        self.batch_size = 100
        self.max_path_length = 250
        self.n_itr = 10
        self.num_envs = 16
        self.M = 60
        self.K = 20
        self.gamma = 0.99
        self.sampling_period = 5


env = PortfolioEnv()
policy = GaussianMLPPolicy(
    name='policy',
    env_spec=env.spec,
    hidden_nonlinearity=tf.nn.relu,
    hidden_sizes=(100, 100),
)

baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = VPG(
    env=env,
    policy=policy,
    load_policy=initial_params_file,
    baseline=baseline,
    batch_size=4000,
    max_path_length=200,
    n_itr=n_itr,
    reset_arg=goal,
    optimizer_args={'init_learning_rate': step_sizes[step_i], 'tf_optimizer_args': {'learning_rate': 0.5 * step_sizes[step_i]}}
)

