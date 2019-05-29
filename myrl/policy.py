
from test_rl.base import Policy, StochasticPolicy
from test_rl.misc import ext
from test_rl.distribution import DiagonalGaussian

import numpy as np
import tensorflow as tf
import time

from collections import OrderedDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class MyModel(Model):
    def __init__(self,
                 dim_output,
                 hidden_sizes,
                 hidden_kernel_initializer='glorot_uniform',
                 hidden_bias_initializer='zeros',
                 output_kernel_initializer='glorot_uniform',
                 output_bias_initializer='zeros',
                 hidden_activation='relu',
                 output_activation='linear',
                 weight_normalization=False):
        super(MyModel, self).__init__()
        self.dim_output = dim_output
        self.hidden_layers = list()
        for dim_h in hidden_sizes:
            self.hidden_layers.append(Dense(dim_h,
                                            kernel_initializer=hidden_kernel_initializer,
                                            bias_initializer=hidden_bias_initializer,
                                            activation=hidden_activation))

        self.output_layer = Dense(dim_output,
                                  kernel_initializer=output_kernel_initializer,
                                  bias_initializer=output_bias_initializer,
                                  activation=output_activation)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        for h_layer in self.hidden_layers:
            x = h_layer(x)
        x = self.output_layer(x)

        return x


class MAMLGaussianMLPPolicy(StochasticPolicy):
    def __init__(self,
                 name,
                 env_spec,
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 min_std=1e-6,
                 std_hidden_nonlinearity=tf.nn.tanh,
                 hidden_nonlinearity='relu',
                 output_nonlinearity='linear',
                 mean_network=None,
                 std_network=None,
                 std_parametrization='exp',
                 grad_step_size=1.0,
                 stop_grad=False,
                 ):
        # assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.input_shape = (None, obs_dim, )
        self.step_size = grad_step_size
        self.stop_grad = stop_grad
        if type(self.step_size) == list:
            raise NotImplementedError

        if mean_network is None:
            self.mean_network = MyModel(dim_output=self.action_dim,
                                        hidden_sizes=hidden_sizes,
                                        hidden_nonlinearity=hidden_nonlinearity,
                                        output_nonlinearity=output_nonlinearity)

            # self.all_params = self.create_MLP(
            #     name='mean_network',
            #     output_dim=self.action_dim,
            #     hidden_sizes=hidden_sizes,
            # )
            # self.input_tensor, _ = self.forward_MLP('mean_network', self.all_params, reuse=None)
            # forward_mean = lambda x, params, is_train: self.forward_MLP('mean_network', params, input_tensor=x, is_training=is_train)[1]
        else:
            raise NotImplementedError

        if std_network is not None:
            raise NotImplementedError
        else:
            if adaptive_std:
                raise NotImplementedError
            else:
                if std_parametrization == 'exp':
                    init_std_param = np.log(init_std)
                elif std_parametrization == 'softplus':
                    init_std_param = np.log(np.exp(init_std) - 1)
                else:
                    raise NotImplementedError


                # self.all_params['std_param'] = make_param_layer(
                #     num_units=self.action_dim,
                #     param=tf.constant_initializer(init_std_param),
                #     name='output_std_param',
                #     trainable=learn_std,
                # )
                # forward_std = lambda x, params: forward_param_layer(x, params['std_param'])

            self.all_param_vals = None

            self._forward = lambda obs, params, is_train: (
                forward_mean(obs, params, is_train), forward_std(obs, params))

            self.std_parametrization = std_parametrization

            if std_parametrization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parametrization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            self._dist = DiagonalGaussian(self.action_dim)

            self._cached_params = {}

            super().__init__(env_spec)

            dist_info_sym = self.dist_info_sym(self.input_tensor, dict(), is_training=False)
            mean_var = dist_info_sym['mean']
            log_std_var = dist_info_sym['log_std']

            self._init_f_dist = tensor_utils.compile_function(
                inputs=[self.input_tensor],
                outputs=[mean_var, log_std_var],
            )
            self._cur_f_dist = self._init_f_dist

    def switch_to_init_dist(self):
        self._cur_f_dist = self._init_f_dist
        self._cur_f_dist_i = None
        self.all_param_vals = None

    def set_init_surr_obj(self, input_list, surr_objs_tensor):
        # set the surrogate objectives used the update the policy. (called by algo.init_opt)
        self.input_list_for_grad = input_list
        self.surr_objs = surr_objs_tensor

    def assign_params(self, tensor_dict, param_values):
        if 'assign_placeholders' not in dir(self):
            self.assign_placeholders = {}
            self.assign_ops = {}
            for key in tensor_dict.keys():
                self.assign_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_ops[key] = tf.assign(tensor_dict[key], self.assign_placeholders[key])

        feed_dict = {self.assign_placeholders[key]: param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_ops, feed_dict)

    def compute_updated_dists(self, samples):
        start = time.time()
        num_tasks = len(samples)
        param_keys = self.all_params.keys()
        update_param_keys = param_keys
        no_update_param_keys = []

        sess = tf.get_default_session()

        obs_list, action_list, adv_list = [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(samples[i], 'observations', 'actions', 'advantages')
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])

        inputs = obs_list + action_list + adv_list

        # to do a second update, replace self.all_params below with the params that were used to collect the policy
        init_param_values = None
        if self.all_param_vals is not None:
            init_param_values = self.get_variable_values(self.all_params)

        step_size = self.step_size
        for i in range(num_tasks):
            if self.all_param_vals is not None:
                self.assign_params(self.all_params, self.all_param_vals[i])

        if 'all_fast_params_tensor' not in dir(self):
            # make computation graph once
            self.all_fast_params_tensor = []
            for i in range(num_tasks):
                gradients = dict(zip(update_param_keys, tf.gradients(self.surr_objs[i], [self.all_params[key] for key in update_param_keys])))
                fast_params_tensor = OrderedDict(zip(update_param_keys, [self.all_params[key] - step_size * gradients[key] for key in update_param_keys]))

                for k in no_update_param_keys:
                    fast_params_tensor[k] = self.all_params[k]
                self.all_fast_params_tensor.append(fast_params_tensor)

        # pull new param vals out of tensorflow, so gradient computation obly once ## first is the vars, second the values
        # these are the updated values of the params after the gradient step
        self.all_param_vals = sess.run(self.all_fast_params_tensor, feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))

        if init_param_values is not None:
            self.assign_params(self.all_params, init_param_values)

        outputs = []
        self._cur_f_dist_i = {}
        inputs = tf.split(self.input_tensor, num_tasks, 0)
        for i in range(num_tasks):
            # TODO: use a placeholder to feed in the params, so that we don't have to recompile every time
            task_inp = inputs[i]
            info, _ = self.dist_info_sym(task_inp, dict(), all_params=self.all_param_vals[i], is_training=False)

            outputs.append([info['mean'], info['log_std']])

        self._cur_fi_dist = tensor_utils.compile_function(inputs=[self.input_tensor], outputs=outputs)

    def dist_info_sym(self, obs_var, state_info_vars=None, all_params=None, is_training=True):
        # This function construct the tf graph, only called during beginning of meta-training
        # obs_var - observation tensor
        # mean_var - tensor for policy mean
        # std_param_var - tensor for policy std before output
        return_params = True
        if all_params is None:
            return_params = False
            all_params = self.all_params

        mean_var, std_param_var = self._forward(obs_var, all_params, is_training)
        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        if self.std_parametrization == 'exp':
            log_std_var = std_param_var
        elif self.std_parametrization == 'softplus':
            log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var)))
        else:
            raise NotImplementedError
        if return_params:
            return dict(mean=mean_var, log_std=log_std_var), all_params
        else:
            return dict(mean=mean_var, log_std=log_std_var)

    def updated_dist_info_sym(self, task_id, surr_obj, new_obs_var, params_dict=None, is_training=True):
        """symbolically create MAML graph, for the meta-optimization, only called at the beginning of meta-training.
        Called more than once if you want to do more than one grad step."""
        old_params_dict = params_dict

        step_size = self.step_size

        if old_params_dict == None:
            old_params_dict = self.all_params

        param_keys = self.all_params.keys()
        update_param_keys = param_keys
        no_update_param_keys = []

        grads = tf.gradients(surr_obj, [old_params_dict[key] for key in update_param_keys])
        if self.stop_grad:
            grads = [tf.stop_gradient(grad) for grad in grads]

        gradients = dict(zip(update_param_keys, grads))
        params_dict = dict(zip(update_param_keys, [old_params_dict[key] - step_size * gradients[key] for key in update_param_keys]))
        for k in no_update_param_keys:
            params_dict[k] = old_params_dict[k]

        return self.dist_info_sym(new_obs_var, all_params=params_dict, is_training=is_training)

