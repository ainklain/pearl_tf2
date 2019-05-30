
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


class ExplorationPolicy(Policy):
    def set_num_steps_total(self, t):
        pass
