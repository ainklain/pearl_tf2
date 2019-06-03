import tensorflow_probability.python.distributions as tfd
import tensorflow_probability as tfp

# Define a single scalar Normal distribution.
dist = tfd.Normal(loc=0., scale=1.)

# Evaluate the cdf at 1, returning a scalar.
dist.cdf(1.)

# Define a batch of two scalar valued Normals.
# The first has mean 1 and standard deviation 11, the second 2 and 22.
dist = tfd.Normal(loc=[1, 2.], scale=[11, 22.])

# Evaluate the pdf of the first distribution on 0, and the second on 1.5,
# returning a length two tensor.
dist.prob([0, 1.5])

# Get 3 samples, returning a 3 x 2 tensor.
dist.sample([3])



import tensorflow as tf
x = tf.Variable([1., 0.2])
with tf.GradientTape(persistent=True) as tape:
    y = x * x
    # z = x + y
    z = x + tf.stop_gradient(y)

grad = tape.gradient(z, x)
grad2 = tape.gradient(z, y)
grad3 = tape.gradient(y, x)
print(grad, grad2, grad3)
del tape
