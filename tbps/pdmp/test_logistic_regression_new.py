import math
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow import keras
from tbnn.pdmp.bps import IterBPSKernel


from tbnn.pdmp.networks import get_network
from tbnn.pdmp import networks

from tbnn.pdmp.model import (plot_density, get_model_state, pred_forward_pass,
                             get_map, get_mle,
                             trace_fn, graph_hmc, nest_concat, set_model_params,
                             build_network,
                             bps_main, hmc_main, nuts_main,
                             pbps_iter_main, cov_pbps_iter_main, get_map_iter,
                             sgld_iter_main, save_map_weights)
#import some helper functions
from tbnn.utils import utils, display, summarise
from sklearn.model_selection import train_test_split

import argparse
import os
import neptune
import sys
import time
import pickle

tfd = tfp.distributions


def get_data(data, split='70_15_15', num_data=1000, num_test=100, random_state=0):
  dataset = utils.load_dataset(data, split)
  # won't use all of the data for now, just using a subset to start off with
  # will just get a subset from the training data
  # X_train, X_test, y_train, y_test = train_test_split(dataset.x_train.astype(np.float32),
  #                                                     dataset.y_train.astype(np.float32),
  #                                                     train_size=num_data,
  #                                                     test_size=num_test,
  #                                                     random_state=random_state)
  # return (X_train, X_test), (y_train, y_test), dataset.dimension_dict
  return (dataset.x_train, dataset.x_test), (dataset.y_train, dataset.y_test), dataset.x_test_orig, dataset.dimension_dict



def examine_rate(model, parent_bnn_neg_joint_log_prob,
                 state, X_train, y_train, out_dir, num_samp=100):
  kernel = IterBPSKernel(
    parent_target_log_prob_fn=parent_bnn_neg_joint_log_prob,
    store_parameters_in_results=True,
    lambda_ref=0.1)
  init_state = [tf.convert_to_tensor(x) for x in state]
  for test_iter in range(0, 10):
    print('eval loop {}'.format(test_iter))
    bps_results = tfp.mcmc.sample_chain(
      num_results=10,
      current_state=init_state,
      return_final_kernel_results=True,
      kernel=kernel)
    samples = bps_results.all_states
    # initialise stafe for next iter
    init_state = [x[-1, ...] for x in samples]
    # final kernel results used to initialise next call of loop
    kernel_results = bps_results.final_kernel_results
    velocity = kernel_results.velocity
    # now iterate over the time steps to evaluate the
    time_dt = tf.constant(0.002, dtype=tf.float32)
    time = tf.Variable(0.0, dtype=tf.float32)
    test = np.zeros(num_samp)
    for i in range(0, num_samp):
      test[i] = kernel.examine_event_intensity(state, velocity, time).numpy()
      time = time + time_dt
    time_arr = np.linspace(0, time_dt.numpy() * num_samp, num_samp)
    plt.figure()
    plt.plot(time_arr, test)
    plt.xlabel('time')
    plt.ylabel('IPP intensity')
    plt.savefig(os.path.join(out_dir, 'ipp_test_{}.png'.format(test_iter)))
    np.save(os.path.join(out_dir, 'conv_time_array_{}.npy'.format(test_iter)), time_arr)
    np.save(os.path.join(out_dir, 'conv_test_array_{}.npy'.format(test_iter)), test)





def classification_accuracy(model, input_, labels, batch_size):
  """get classification accuracy
  Args:
    model (keras.Model):
      neural network model to evaluate
   input_ (np.array):
      input data to be passed through the model
    labels (np.array):
      true label data corresponding to the supplied input_
  Returns:
    classification accuracy
  """
  classification_array, correct_labels = classify(model, input_,
                                                  labels, batch_size)
  # now find the accuracy
  accuracy = accuracy_score(correct_labels,
                            classification_array,
                            normalize=True)
  return accuracy


def classify(model, input_, labels, batch_size):
  """Helper func. classify all the examples
  Args:
  input_ (np.array):
    input data to evaluate
  labels (np.array):
    true labels corresponding to the inputs
  Returns:
  classification_array (np.array):
    list of what the classifier actually predicted
  correct_prediction_eval (np.array):
    list of what the correct prediction labels should be
  """
  data = tf.data.Dataset.from_tensor_slices(
    (input_, labels))
  # batch testing data based on prediction type and the no. test samples
  # send the 2 times the batch size across
  # if the GPU can handle the original batch size for training,
  # then shouldn't have an issue with double for testing
  data = data.batch(np.int(2 * batch_size))
  # forming lists to save the output
  classification_list = []
  label_list = []
  # classify each batch and store the results
  for input_batch, labels_batch in data:
    output_ = model(input_batch)
    #print('output_ = {}'.format(output_))
    classification_list.extend(np.argmax(output_, axis=1))
    label_list.extend(np.argmax(labels_batch, axis=1))
  classification_array = np.array(classification_list)
  correct_prediction = np.array(label_list)
  #Wprint('classification_array = {}'.format(classification_array))
  #print('correct_prediction   = {}'.format(correct_prediction))
  return classification_array, correct_prediction


def neg_log_likelihood(model, likelihood_fn, X, y):
  logits = model(X)
  log_likelihood_dist = likelihood_fn(logits, scale=0.2)
  # add the log likelihood now
  lp = tf.reduce_sum(log_likelihood_dist.log_prob(y))
  return -1.0 * lp


def bnn_neg_joint_log_prob_fn(model, likelihood_fn, dataset_iter):
  X, y = dataset_iter.next()
  def _fn(*param_list):
    with tf.name_scope('bnn_joint_log_prob_fn'):
      # set the model params
      m = set_model_params(model, param_list)
      # print('current  model params ttttt= {}'.format(model.layers[1].kernel))
      # neg log likelihood of predicted labels
      neg_ll = neg_log_likelihood(model, likelihood_fn, X, y)
      # now get the losses from the prior (negative log prior)
      # these are stored within the models `losses` variable
      neg_lp = tf.reduce_sum(model.losses)
      # add them together for the total loss
      return neg_ll + neg_lp
  return _fn


def main(args):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  print('gpus = {}'.format(gpus))
  #tf.config.experimental.set_virtual_device_configuration(
  #  gpus[0],
  #  [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  (X_train, X_test), (y_train, y_test), X_test_orig, data_dimension_dict = get_data(args.data)
  training_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  print('training size = {}'.format(training_data.cardinality().numpy()))
  # shuffle the training data set, repeat and batch it
  training_data = training_data.shuffle(X_train.shape[0]).batch(args.batch_size)
  print('training size = {}'.format(training_data.cardinality().numpy()))
  # make it repeat indefinitely
  training_data = training_data.repeat()
  # now create an iter object of it
  training_iter = iter(training_data)
  # get single sample to build the model
  model = get_network(args.network,
                      X_train[0].shape,
                      y_train[0].size,
                      prior=1.0)
  likelihood_fn = networks.get_likelihood_fn(args.likelihood)
  print(model)
  print(model.summary())
  for var in model.trainable_variables:
    print('var name {}, var shape = {}, var_size = {}'.format(
      var.name, var.shape, tf.reduce_prod(var.shape)))
  print('model type = {}'.format(model))
  bnn_neg_joint_log_prob = bnn_neg_joint_log_prob_fn(model,
                                                     likelihood_fn,
                                                     training_iter)
  # get the initial state for obtaining MAP estimate.
  # This can just be the getting initial values from the model we defined
  initial_state = get_model_state(model)
  #print('initial_state = {}'.format(initial_state))
  print('Dataset = {}'.format(training_iter))
  if args.map_path == None:
    map_start = time.time()
    map_initial_state = get_map(bnn_neg_joint_log_prob,
                                initial_state, model,
                                num_iters=args.map_iters,
                                save_every=10000)
    map_end = time.time()
    print('time to find MAP estimate = {}'.format(map_end - map_start))
    print('map_initial_state = {}'.format(map_initial_state))
    accuracy = classification_accuracy(model, X_test, y_test, args.batch_size)
    print('Test accuracy from MAP = {}'.format(accuracy))
    # save the MAP weights
    save_map_weights(map_initial_state, args.out_dir)
    pred = pred_forward_pass(model, map_initial_state,
                             X_train.astype(np.float32))

    plt.plot(X_train, pred, color='k')
    print(X_train.shape)
    print(y_train.shape)
    plt.scatter(X_train, y_train, color='b', alpha=0.25)
    plt.savefig(os.path.join(args.out_dir, 'pred_map.png'))

  else:
    with open(args.map_path, 'rb') as f:
      map_initial_state = pickle.load(f)

  # examine_rate(model, bnn_neg_joint_log_prob, map_initial_state,
  #              X_train, y_train, args.out_dir)
  # now train MCMC method if specified
  # number of samples available for training
  data_size = X_train.shape[0]
  if args.bps:
    bps_main(model, args.ipp_sampler, args.likelihood, args.ref, args.num_results,
             args.num_burnin, args.out_dir,
             bnn_neg_joint_log_prob, map_initial_state,
             X_train, y_train, X_test, y_test, data_size, data_size, data_dimension_dict,
             plot_results=~args.no_plot)

#   if args.bps:
#     bps_main(model, args.ipp_sampler, args.ref, args.num_results,
#              args.num_burnin, args.out_dir,
#              bnn_neg_joint_log_prob, map_initial_state,
#              X_train, X_test, X_test, y_test, X_test_orig,
#              args.batch_size, data_size, data_dimension_dict,
#              plot_results=~args.no_plot)

def check_cmd_args(args):
  """check all the commandline arguments are valid"""
  # check network argument
  networks.check_network_arg(args.network)
  # now the likelihood
  networks.check_likelihood_arg(args.likelihood)
  # check prior is ok, and format if needed
  args.prior = networks.check_format_prior_arg(args.prior)
  # lets check that the directories suplied exist, and if they don't,
  # lets make them
  utils.check_or_mkdir(args.out_dir)
  return args


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='test_conv_new',
                                   epilog=main.__doc__,
                                   formatter_class=
                                   argparse.RawDescriptionHelpFormatter)
  parser.add_argument('network', type=str,
                      help='network type')
  parser.add_argument('likelihood', type=str,
                      help='type of likelihood function to use')
  parser.add_argument('prior', type=str,
                      help='prior to be used')
  parser.add_argument('--out_dir', type=str, default='./out',
                      help='out directory where data is saved')
  parser.add_argument('--opt', type=str, default='adam',
                      help='optimizer to be used')
  parser.add_argument('--batch_size', type=int, default=100,
                      help='Number of samples per batch')
  parser.add_argument('--data', type=str, default='mnist_im',
                      help='data set to use')
  parser.add_argument('--ref', type=float, default=1.0,
                      help='lambda for refresh poisson process')
  parser.add_argument('--map_iters', type=int, default=20000,
                      help='number of iterations for map estimate')
  parser.add_argument('--map_path', type=str, default=None,
                      help='path to load map weights')
  parser.add_argument('--bps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run BPS')
  parser.add_argument('--cov_pbps', type=bool, default=False, nargs='?',
                      const=True, help='whether to run Cov precond. BPS')
  parser.add_argument('--pbps', type=bool, default=False, nargs='?',
                      const=True, help='sbps')
  parser.add_argument('--sgld', type=bool, default=False, nargs='?',
                      const=True, help='whether to run sgld')
  parser.add_argument('--ipp_sampler', type=str, default='adaptive', nargs='?',
                      help='type of sampling scheme for event IPP')
  parser.add_argument('--hmc', type=bool, default=False, nargs='?',
                      const=True, help='whether to run HMC')
  parser.add_argument('--num_results', type=int, default=100,
                      help='number of sample results')
  parser.add_argument('--num_burnin', type=int, default=100,
                      help='number of burnin samples')
  description_help_str = ('experiment description'
                          '(place within single quotes \'\'')
  parser.add_argument('--description', type=str, default='test-logistic',
                      nargs='?', help=description_help_str)
  parser.add_argument('--exp_name', type=str, default='test-conv', nargs='?',
                        help='name of experiment (usually don\'t have to change)')
  parser.add_argument('--no_log', type=bool, default=False,
                      nargs='?', const=True,
                      help='whether should skip logging to neptune or not')
  parser.add_argument('--no_plot', type=bool, default=False,
                      nargs='?', const=True,
                      help='whether should skip plotting or getting pred metrics')
  args = parser.parse_args(sys.argv[1:])
  # now lets check all the arguments
  args = check_cmd_args(args)
  # if args.bps == False and args.hmc == False and args.nuts == False:
  #   raise ValueError('Either arg for BPS, HMC or NUTS must be supplied')
  # if we are logging info to neptune, initialise the logger now
  exp_params = vars(args)
  print('args = {}'.format(args))
  exp_tags = [key for key, x in exp_params.items() if (isinstance(x, bool) and (x == True))]
  if(not args.no_log):
    print('logging to neptune')
    neptune.init('ethangoan/{}'.format(args.exp_name))
    neptune.create_experiment(name='test_conv',
                              params=exp_params,
                              tags=exp_tags,
                              upload_source_files=[args.config, './test_conv.py'])
  main(args)
