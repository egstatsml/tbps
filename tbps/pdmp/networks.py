"""networks.py

Code to make common networks
"""
import numpy as np
import numbers
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Conv2D,  MaxPool2D, Flatten,
                                     AvgPool2D, GlobalAveragePooling2D, ReLU, Input,
                                     BatchNormalization, Layer, Add, InputLayer)
from tensorflow.keras import Model, Input
import tensorflow as tf
from tensorflow_addons.layers import FilterResponseNormalization, TLU
import tensorflow_probability as tfp
from tbnn.pdmp.resnet import ResNetLayer, Swish, make_resnet
from tbnn.pdmp.frn import FRN
tfd = tfp.distributions


class Swish(Layer):
  def __init__(self):
    super(Swish, self).__init__()

  def call(self, inputs):
    return keras.activations.swish(inputs)




# permitted values for the different arguments specified by cmdline args
VALID_NETWORKS = ['lenet5', 'resnet20', 'kaggle', 'small_cnn', 'small_regression',
                  'retinopathy', 'uci_mlp', 'med_mnist', 'cifar_alexnet']
VALID_PRIOR_STRS = ['fan_in']
VALID_LIKELIHOOD = ['normal', 'bernoulli', 'categorical']

class InvalidLikelihood(ValueError):
  def __init__(self, arg):
    print('Invalid likelihood function specified in config file')
    print('Supplied {}, available loss functions are {}'.format(
      arg, ['normal', 'bernoulli', 'categorical']))
    ValueError.__init__(self)


def check_network_arg(arg):
  """checks the command line argument for the network string is valid"""
  if arg.lower() not in VALID_NETWORKS:
    raise ValueError(
      'Invalid network type supplied. Expect one of {}, got {}'.format(
        VALID_NETWORKS, arg))


def check_format_prior_arg(arg):
  """checks the command line argument for the prior and formats if needed

  Will want to format it if the supplied argument is a number, as the arg will
  be interpretted from the commandline as a string. If it is a number, convert
  it to a float, otherwise just make sure it is valid
  """
  try:
    prior = float(arg)
    # if it made it past here, then it is a float
    # do some quick error checking to make sure it isn't negative
    if prior < 0:
      raise ValueError(
        'Invalid prior specified. If a float, must be positive. got {}'.format(
          prior))
  except ValueError:
    # just check the valid string
    if arg.lower() not in VALID_PRIOR_STRS:
      raise ValueError(
        'Invalid prior specified. If a str, must be one of {}. got {}'.format(
          VALID_PRIOR_STRS, prior))
    prior = arg
  return prior


def check_likelihood_arg(arg):
  """checks the likelihood argument supplied is valid"""
  if arg.lower() not in VALID_LIKELIHOOD:
    raise ValueError(
      'Invalid likelihood specified. Expected one of{}. got {}'.format(
        VALID_LIKELIHOOD, arg))


def get_prior_neg_log_prob_fn(prior,
                              kernel_dims=None,
                              use_bias=True,
                              is_bias=False):
  """will get the prior functions needed for this layer

  Will handle two types of priors:
    1) Gaussian with all same variance
    2) Gaussian with variance based on fan in dimensions

  The option to allow for fan in dimensions is to allow for models
  which want to set the variance parameter to encourage the output
  variance to be 1.0.

  Args:
    prior (float or str):
      If the prior argument is a float, then will return a Gaussian with a
      variance set to this value. If it is a string saying "fan_in", then
      will scale the  variance such that the output has unit variance
      (under the prior).
    kernel_dims (list(int)):
      dimensions of the kernel for the layer. If it is a a conv layer, will have
      four dimensions, if dense will have two. These are only to be used if
      a prior to scale the output variance is needed.
    use_bias (bool):
      whether the bias has been included or not. Is needed if aiming to scale
      the output variance for a specific layer.
    is_bias (bool):
      boolean to say whether this is the bias variable or not.
      if it is the bias variable, but `use_bias` is false, will just
      set the prior for this variable to `None`

  Returns:
    prior negative log probability function as a specified from tfp.

  Raises:
    Not implemented error for scaling vartiance. Starting with simple
    constant variance and will move to the add scaling in a bit. Check
    my notes for more info.
  """
  # check if the specified prior is an float (or int) or a string to say we
  # should wither have a constant variance.
  if isinstance(prior, numbers.Number):
    # if this is the prior for the bias variable, but we aren't using the bias,
    # then we will just return `None`
    if(is_bias and not use_bias):
      return None
    # otherwise, is either the kernel (which we should always have),
    # or is the bias and we are actually using it
    #
    # then we should just have constant variance
    # lets cast the this value to a tf float32
    p_scale = tf.cast(prior, dtype=tf.float32)
    prior_fn = tfd.Normal(loc=0., scale=p_scale)
    def prior_constant_var_neg_log_prob(x):
      log_prob = tf.reduce_sum(prior_fn.log_prob(x))
      # return negative log prob
      return -1.0 * log_prob
    return prior_constant_var_neg_log_prob
  else:
    # otherwise need to scale the variance.
    raise NotImplementedError('Currently only support constant var')


def get_likelihood_fn(likelihood_str):
  """get likelihood for the current model

  For the MCMC models it should be the name of a TFP distribution.
  This method checks that the likelihood supplied is valid, and then
  sets the returns the corresponding likelihood fn.


  Args:
    output_dims (str):
      dimension of the output
    likelihood_str (str):
      string to specify distribution used to model the likelihood

  Returns:
   the input argument if it is of a valid form

   Raises:
    `ValueError()` if incorrect likelihood str
  """
  likelihood_str = likelihood_str.lower()
  # check the likelihood string
  check_likelihood_arg(likelihood_str)
  # normal dist
  if(likelihood_str == 'normal'):
    dist = tfd.Normal
  # binary classification
  elif(likelihood_str == 'bernoulli'):
    dist = tfd.Bernoulli
  # categorical classification
  elif(likelihood_str == 'categorical'):
    dist = tfd.OneHotCategorical
  else:
    raise ValueError(('Problem with the likelihood specified. '
                      'Should not of made it here, making it here means '
                      'that the error checking done to see if your '
                      'likelihood is valid is dang broke. '
                      'Expected something like {}, but got {}').format(
                        VALID_LIKELIHOOD, likelihood_str))
  return dist


def get_lenet_5(input_dims, num_classes,
                use_bias=True, prior=1.0,
                activation='relu'):
  print('input dims = {}'.format(input_dims))
  # build the model
  inputs = Input(input_dims)
  x = Conv2D(6, kernel_size=5, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(inputs)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  x = Conv2D(16, kernel_size=5, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  x = Conv2D(120, kernel_size=5, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = Flatten()(x)
  x = Dense(84,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(x)
  outputs = Dense(num_classes,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias, is_bias=True),
                  activation=None)(x)
  lenet5 = Model(inputs, outputs)
  return lenet5


def get_small_cnn(input_dims, num_classes,
                  use_bias=True, prior=1.0,
                  activation='relu'):
  # build the model
  inputs = Input(input_dims)
  x = Conv2D(16, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(inputs)
  x = Conv2D(32, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  x = Conv2D(32, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = Conv2D(64, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  x = Flatten()(x)
  x = Dense(256,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(x)
  outputs = Dense(num_classes,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias, is_bias=True),
                  activation=None)(x)
  small_cnn = Model(inputs, outputs)
  return small_cnn

def get_small_regression(input_dims, out_dims,
                         use_bias=True, prior=1.0,
                         activation='relu'):
  inputs = Input(input_dims)
  x = Dense(100,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(inputs)
  # x = Dense(50,
  #           kernel_regularizer=get_prior_neg_log_prob_fn(
  #             prior, use_bias=use_bias),
  #           use_bias=use_bias,
  #           bias_regularizer=get_prior_neg_log_prob_fn(
  #             prior, use_bias=use_bias, is_bias=True),
  #           activation=activation)(x)
  x = Dense(20,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(x)
  outputs = Dense(out_dims,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias, is_bias=True),
                  activation=None)(x)
  small_regression = Model(inputs, outputs)
  return small_regression


def get_uci_mlp(input_dims, out_dims,
                use_bias=True, prior=1.0,
                activation='relu'):
  inputs = Input(input_dims)
  x = Dense(50,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(inputs)
  outputs = Dense(2,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias, is_bias=True),
                  activation=None)(x)
  uci_mlp = Model(inputs, outputs)
  return uci_mlp


def get_med_mnist(input_dims, num_classes,
                  use_bias=True, prior=1.0,
                  activation='relu'):
  print('input dims = {}'.format(input_dims))
  # build the model
  inputs = Input(input_dims)
  x = Conv2D(6, kernel_size=5, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(inputs)
  x = AvgPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  x = Conv2D(16, kernel_size=5, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = AvgPool2D(pool_size=[2, 2], strides=[2, 2], padding='SAME')(x)
  x = Flatten()(x)
  x = Dense(120,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(x)
  x = Dense(84,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(x)
  outputs = Dense(num_classes,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias, is_bias=True),
                  activation=None)(x)
  med_mnist = Model(inputs, outputs)
  return med_mnist


def get_cifar_alexnet(input_dims, num_classes,
                      use_bias=True, prior=1.0,
                      activation='swish'):
  print('input dims = {}'.format(input_dims))
  # build the model
  inputs = Input(input_dims)
  x = Conv2D(64, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(inputs)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='VALID')(x)
  x = Conv2D(128, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='VALID')(x)
  x = Conv2D(256, kernel_size=2, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = Conv2D(128, kernel_size=2, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = Conv2D(64, kernel_size=2, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = Flatten()(x)
  x = Dense(256,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(x)
  x = Dense(256,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(x)
  outputs = Dense(num_classes,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias, is_bias=True),
                  activation=None)(x)
  cifar_alexnet = Model(inputs, outputs)
  return cifar_alexnet


def get_retinopathy(input_dims, num_classes,
                    use_bias=True, prior=1.0,
                    activation='swish'):
  print('input dims = {}'.format(input_dims))
  # build the model
  inputs = Input(input_dims)
  x = Conv2D(32, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(inputs)
  x = Conv2D(32, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='VALID')(x)
  x = Conv2D(32, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = Conv2D(32, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='VALID')(x)
  x = Conv2D(16, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = Conv2D(16, kernel_size=3, padding='SAME',
             kernel_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias),
             use_bias=use_bias,
             bias_regularizer=get_prior_neg_log_prob_fn(
               prior, use_bias=use_bias, is_bias=True),
             activation=activation)(x)
  x = MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='VALID')(x)
  x = Flatten()(x)
  x = Dense(128,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(x)
  outputs = Dense(num_classes,
                  kernel_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias),
                  use_bias=use_bias,
                  bias_regularizer=get_prior_neg_log_prob_fn(
                    prior, use_bias=use_bias, is_bias=True),
                  activation=None)(x)
  retinopathy = Model(inputs, outputs)
  return retinopathy


def get_linear(input_dims, out_dims,
               use_bias=True, prior=1.0,
               activation='None'):
  inputs = Input(input_dims)
  x = Dense(50,
            kernel_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias),
            use_bias=use_bias,
            bias_regularizer=get_prior_neg_log_prob_fn(
              prior, use_bias=use_bias, is_bias=True),
            activation=activation)(inputs)


def get_network(network, input_dims, output_dims,  prior,
                use_bias=True, activation='relu'):
  if network == 'lenet5':
    model = get_lenet_5(input_dims,
                        output_dims,
                        prior,
                        use_bias)
  elif network == 'resnet20':
    model = make_resnet(output_dims,
                        20,
                        input_dims)
  elif network == 'small_cnn':
    model = get_small_cnn(input_dims,
                           output_dims,
                           prior,
                           use_bias)
  elif network == 'small_regression':
    model = get_small_regression(input_dims,
                                 output_dims,
                                 prior,
                                 use_bias)
  elif network == 'uci_mlp':
    model = get_uci_mlp(input_dims,
                        2,
                        prior,
                        use_bias)
  elif network == 'med_mnist':
    model = get_med_mnist(input_dims,
                          7,
                          prior,
                          use_bias)
  elif network == 'cifar_alexnet':
    model = get_cifar_alexnet(input_dims,
                              10,
                              prior,
                              use_bias)
  elif network == 'retinopathy':
    model = get_retinopathy(input_dims,
                            output_dims,
                            prior,
                            use_bias)
  else:
    raise ValueError('Invalid network supplied, got {}'.format(network))
  return model
