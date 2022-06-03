import abc
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
import sys
import collections

from tbnn.pdmp.poisson_process import SBPSampler, AdaptiveSBPSampler
from tbnn.pdmp.utils import compute_dot_prod
from tbnn.nn.mlp import MLP
from tbnn.pdmp.hull_tf import sample_poisson_thinning
