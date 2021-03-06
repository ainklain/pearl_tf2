
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
