# from mtrl.misc import autoargs


class Algorithm(object):
    pass


class RLAlgorithm(Algorithm):
    def train(self):
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