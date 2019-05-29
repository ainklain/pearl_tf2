from test_rl.misc import ext, tensor_utils

import tensorflow as tf
import scipy.optimize
import time

class LbfgsOptimizer:
    def __init__(self, name, max_opt_itr=20, callback=None):
        self._name = name
        self._max_opt_itr =max_opt_itr
        self._opt_fun = None
        self._target = None
        self._callback = callback

    def update_opt(self, loss, target, inputs, extra_inputs=None, *args, **kwargs):
        self._target = target

        def get_opt_output():
            return [loss value and gradient values]

        if extra_inputs is None:
            extra_inputs = list()

        self._opt_fun = ext.lazydict(
            f_loss=lambda: loss_function(inputs + extra_inputs, loss),
            f_opt=lambda: grad_function(inputs=inputs + extra_inputs, outputs=get_opt_output()))

    def loss(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = list()
        return self._opt_fun['f_loss'](*(list(inputs) + list(extra_inputs)))

    def optimize(self, inputs, extra_inputs=None):
        f_opt = self._opt_fun['f_opt']

        if extra_inputs is None:
            extra_inputs = list()

        def f_opt_wrapper(flat_params):
            self._target.set_param_values(flat_params, trainable=True)
            ret = f_opt(*inputs)
            return ret

        itr = [0]
        start_time = time.time()

        if self._callback:
            def opt_callback(params):
                loss = self._opt_fun['f_loss'](*(inputs + extra_inputs))
                elapsed = time.time() - start_time
                self._callback(dict(
                    loss=loss,
                    params=params,
                    itr=itr[0],
                    elapsed=elapsed,))
                itr[0] += 1
        else:
            opt_callback = None

        scipy.optimize.fmin_l_bfgs_b(
            func=f_opt_wrapper,
            x0=self._target.get_param_values(trainable=True),
            maxiter=self._max_opt_itr,
            callback=opt_callback,
        )



class FirstOrderOptimizer:
    def __init__(self,
                 tf_optimizer_cls=None,
                 tf_optimizer_args=None,
                 max_epochs=1000,
                 tolerance=1e-6,
                 batch_size=32,
                 callback=None,
                 verbose=False,
                 init_learning_rate=None,
                 **kwargs):

        self._opt_fun = None
        self._target = None
        self._callback = callback
        if tf_optimizer_cls is None:
            tf_optimizer_cls = tf.optimizers.Adam
        if tf_optimizer_args is None:
            tf_optimizer_args = dict(learning_rate=1e-3)
        self.learning_rate = tf_optimizer_args['learning_rate']
        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._init_tf_optimizer = None
        if init_learning_rate is not None:
            init_tf_optimizer_args = dict(learning_rate=init_learning_rate)
            self._init_tf_optimizer
