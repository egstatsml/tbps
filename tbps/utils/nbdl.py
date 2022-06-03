import os
import sys
import math
import einops
import tensorflow as tf
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def load_data(data_str, data_dir='~/exp_data/neurips_bdl_comp_data/data'):
  """ load in the data we need.

  Args:
    data_str (str):
      string of dataset to use
    data_dir (str):
      root directory holding all the data

  Returns:
    train and test sets for the given dataset.

  Raises:
    IOError if invalid dataset string supplied or the data_dir
    supplied is no good.
    (ie. when trying to load dataset that doesn't exist)
  """
  data = np.load(os.path.join(data_dir, data_str + '_anon.npz'))
  x_train = data["x_train"]
  y_train = data["y_train"]
  x_test = data["x_test"]
  y_test = data["y_test"]
  # if the data is the retinopathy dataset, need to change the order of the data
  # by reshaping it
  if data_str in 'retinopathy':
    x_train_ = einops.rearrange(x_train, "n h w c -> n c h w")
    x_test_ = einops.rearrange(x_test, "n h w c -> n c h w")
 return x_train, y_train, x_test, y_test
