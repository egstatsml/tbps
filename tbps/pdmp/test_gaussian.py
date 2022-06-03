import math
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tbnn.pdmp.bps import BPSKernel
from tbnn.pdmp.poisson_process import AdaptiveSBPSampler
from tbnn.pdmp.poisson_process import SBPSampler, PSBPSampler, AdaptivePSBPSampler, AdaptiveSBPSampler

from tbnn.pdmp.model import (plot_density, get_model_state, pred_forward_pass,
                             get_map, get_mle, get_model_state,
                             trace_fn, graph_hmc, nest_concat, set_model_params,
                             build_network,
                             bps_main, cov_pbps_main, hmc_main, nuts_main, pbps_main,
                             get_map_iter)
#import some helper functions
from tbnn.utils import utils, display, summarise
from sklearn.model_selection import train_test_split

from tbnn.pdmp.bps import BPSKernel, IsotropicGaussianBPSKernel

import argparse
import os
import sys
import neptune

tfd = tfp.distributions
from tbnn.embedded_vi.embedded_vi import EmbeddedVIKernel, DenseReparameterizationMAP
from tbnn.vi import utils as vi_utils


# def neg_log_prob_fn():
#   dist = tfd.Normal(loc=[0.0, 0.0], scale=1.0)
#   def _fn(*args):
#     with tf.name_scope('og_prob_fn'):
#       lp = tf.reduce_sum(dist.log_prob(args))
#       return -1.0 * lp
#   return _fn


def neg_log_prob_fn():
  scale = np.array([[1.0, 0.75], [0.75, 1.0]])
  dist = tfd.Normal(loc=[0.0, 0.0], scale=scale)
  def _fn(*args):
    with tf.name_scope('og_prob_fn'):
      lp = tf.reduce_sum(dist.log_prob(args))
      return -1.0 * lp
  return _fn



def main(args):
  gpus = tf.config.list_physical_devices('GPU')
  print('gpus = {}'.format(gpus))
  tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)])
  # get the dist
  neg_lp = neg_log_prob_fn()
  kernel = BPSKernel(
    target_log_prob_fn=neg_lp,
    store_parameters_in_results=True,
    ipp_sampler=SBPSampler,
    batch_size=-1,
    data_size=-1,
    lambda_ref=args.ref)
  # kernel = IsotropicGaussianBPSKernel(
  #   target_log_prob_fn=neg_lp,
  #   store_parameters_in_results=True,
  #   lambda_ref=args.ref)
  init_state = [tf.convert_to_tensor(np.array([0.7, -1.0]).astype(np.float32))]
  # creating the trace function
  trace_fn = lambda _, pkr: pkr.acceptance_ratio
  bps_results, acceptance_ratio = graph_hmc(
    num_results=args.num_results,
    current_state=init_state,
    num_steps_between_results=0,
    kernel=kernel,
    trace_fn=trace_fn)
  bps_chain = np.array(bps_results)
  np.save('gaussian.npy', bps_chain)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='test_bnn',
                                   epilog=main.__doc__,
                                   formatter_class=
                                   argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--ref', type=float, default=1.0,
                      help='lambda for refresh poisson process')
  parser.add_argument('--num_results', type=int, default=100,
                      help='number of sample results')

  args = parser.parse_args(sys.argv[1:])
  main(args)
