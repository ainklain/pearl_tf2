# from mtrl.misc import autoargs


class Algorithm(object):
    pass


class RLAlgorithm(Algorithm):
    def train(self):
        raise NotImplementedError


class Policy:
    def __init__(self, env_spec):
        self._env_spec = env_spec

    def get_action(self, observation):
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def recurrent(self):
        return False

    def log_diagnostics(self, paths):
        pass

    @property
    def state_info_keys(self):
        return list()

    def terminate(self):
        pass


class StochasticPolicy(Policy):
    @property
    def distribution(self):
        raise NotImplementedError

    def dist_info_sym(self, obs_var, state_info_vars):
        raise NotImplementedError

    def dist_info(self, obs, state_infos):
        raise NotImplementedError



class Baseline(object):
    def __init__(self, env_spec):
        self._mdp_spec = env_spec

    @property
    def algorithm_parallelized(self):
        return False

    def get_param_values(self):
        raise NotImplementedError

    def set_param_values(self, val):
        raise NotImplementedError

    def fit(self, paths):
        raise NotImplementedError

    def predict(self, path):
        raise NotImplementedError

    # @classmethod
    # @autoargs.add_args
    # def add_args(cls, parser):
    #     pass

    @classmethod
    def new_from_args(cls, args, mdp):
        pass

    def log_diagnostics(self, paths):
        pass


class Distribution(object):
    @property
    def dim(self):
        raise NotImplementedError

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        raise NotImplementedError

    def kl(self, old_dist_info, new_dist_info):
        raise NotImplementedError

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        raise NotImplementedError

    def entropy(self, dist_info):
        raise NotImplementedError

    def log_likelihood_sym(self, x_var, dist_info_vars):
        raise NotImplementedError

    def log_likelihood(self, xs, dist_info):
        raise NotImplementedError

    @property
    def dist_info_specs(self):
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        return [k for k, _ in self.dist_info_specs]


class Space:
    def sample(self, seed=0):
        raise NotImplementedError

    def contains(self, x):
        raise NotImplementedError

    def flatten(self, x):
        raise NotImplementedError

    def unflatten(self, x):
        raise NotImplementedError

    def flatten_n(self, xs):
        raise NotImplementedError

    @property
    def flat_dim(self):
        raise NotImplementedError

    def new_tensor_variable(self, name, extra_dims):
        raise NotImplementedError