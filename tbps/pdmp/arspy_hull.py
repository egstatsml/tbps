from numpy import asarray, isinf, isnan, spacing as eps, log, exp, cumsum
from arspy.probability_utils import exp_normalize
import numpy as np

import matplotlib.pyplot as plt
import random
import string

__all__ = (
  "HullNode",
  "compute_hulls",
  "evaluate_hulls",
  "sample_upper_hull",
  "compute_segment_log_prob",
)

def sample_poisson_thinning(hull_samples, time):
  #print('hull_samples = {}'.format(hull_samples))
  #print('time = {}'.format(np.sort(time)))
  S = np.sort(np.array(time).ravel())
  fS = np.array(hull_samples).ravel()
  upper_hulls, _ = compute_hulls(S, fS, domain=[0.0, 10.0])
  hull = Hull(upper_hulls)
  sample_time = hull.sample_poisson()
  # find the value of the hull at this location
  sample_hull = hull.eval_hull(sample_time)
  # def random_string_generator(str_size, allowed_chars=string.ascii_letters):
  #   return ''.join(random.choice(allowed_chars) for x in range(str_size))
  # plot_str = random_string_generator(12)
  # plt.figure()
  # #plt.plot(S_hi, fS_hi, c='b', alpha=0.5, linewidth=3, label='target')
  # #plt.scatter(S, fS, alpha=0.25, c='b', label='samples', s=150)
  # for node in hull.hull_list[:-1]:
  #   plt.plot([node.left, node.right],
  #            [node.left * node.m + node.b, node.right * node.m + node.b],
  #            alpha=1.0, c='r', linestyle='--')
  #   plt.scatter(np.array([node.left, node.right]),
  #               np.array([node.left * node.m + node.b, node.right * node.m + node.b]),
  #               alpha=0.5, c='r')
  # plt.plot([hull.hull_list[-1].left, hull.hull_list[-1].right],
  #          [hull.hull_list[-1].left * hull.hull_list[-1].m + hull.hull_list[-1].b,
  #           hull.hull_list[-1].right * hull.hull_list[-1].m + hull.hull_list[-1].b],
  #          c='r', alpha=1.0, linestyle='--', label='envelope')
  # # plt.scatter(np.array([hull.hull_list[-1].right]),
  # #             np.array([hull.hull_list[-1].right * hull.hull_list[-1].m + hull.hull_list[-1].b]),
  # #             c='r', alpha=0.5)
  # plt.legend(loc=0)
  # plt.savefig('adaptive_test/{}_hull.png'.format(plot_str))
  # plt.clf()
  # try:
  #   time, integrated = hull.eval_integrated()
  #   inv_time, inv_integrated = hull.eval_inverse_integrated(time=integrated)
  #   plt.subplot(211)
  #   plt.plot(inv_time, inv_integrated, label='inverse')
  #   plt.subplot(212)
  #   plt.plot(time, integrated, label='integrated')
  #   plt.savefig('adaptive_test/{}.png'.format(plot_str))
  #   plt.close()
  # except:
  #   pass
  # print('this is sample {}'.format(plot_str))
  return sample_time, sample_hull



class Hull(object):
  """a class that contains all the hull nodes

  Is used for sampling from the Hull for a Poisson Process
  """
  def __init__(self, hull_list):
    self.hull_list = hull_list
    # print('\nHull items')
    # for i in range(0, len(hull_list)):
    #   print(self.hull_list[i])
    # now want to update the domain and ranges of the intgrated hulls
    constant = self.hull_list[0].evaluate_integrated(self.hull_list[0].left)
    for i in range(0, len(self.hull_list)):
      integrated_lower = self.hull_list[i].evaluate_integrated(self.hull_list[i].left)
      integrated_upper = self.hull_list[i].evaluate_integrated(self.hull_list[i].right)
      # if(i > 0):
      #   constant -= integrated_lower
      # print('constant = {}'.format(constant))
      # print('lower = {}'.format(integrated_lower))
      # print('constant - lower = {}'.format(constant - integrated_lower))
      self.hull_list[i].constant = constant - integrated_lower
      # now set the range for the integrated rate and the
      # domain for the inverse of the integrated rate, which will
      # be the same
      self.hull_list[i].integrated_range_lower = self.hull_list[i].constant
      self.hull_list[i].integrated_range_upper = integrated_upper + self.hull_list[i].constant
      self.hull_list[i].inverse_integrated_domain_lower =  constant
      self.hull_list[i].inverse_integrated_domain_upper = integrated_upper + self.hull_list[i].constant
      # now add increment the constant term, which is the difference
      # established by the integral of this hull
      # print('hull {}, int_range = ({}, {}], int_domain = ({}, {}]'.format(i, self.hull_list[i].integrated_range_lower,
      #                                                                     self.hull_list[i].integrated_range_upper,
      #                                                                     self.hull_list[i].integrated_domain_lower,
      #                                                                     self.hull_list[i].integrated_domain_upper))
      constant += integrated_upper - integrated_lower

  def sample_poisson(self):
    """ Performs sampling from the poisson process

    To sample using the inversion method, need
    to find
    t = t + \Lambda^{-1}(E + \Lambda(t))

    where E ~ exponential
    For our scenario, we are only trying to find the first sample
    in a poisson process, so T initially equals zero.
    Since our original hull (\lambda) is piecewise linear, that means
    that \Lambda is piewcewise quadratic and each segment is represented as
    \Lambda(t) = (1/2) * a * t^2 + b * t
    meaning that \Lambda(0) = 0, so we don't need to evaluate this, we only
    need to evaluate the inverse w.r.t E.
    """
    E_sample = np.random.exponential(1.0)
    # now need to find which segment E is within the range of
    # for the inverse function
    less_than_upper = [E_sample <= x.inverse_integrated_domain_upper for x in self.hull_list]
    # now find the first hull segment where our exponential variable is
    # greater than the lower bound of the range, as this will be the one
    # where we need to sample from inverse
    try:
      hull_segment_idx = np.where(less_than_upper)[0][0]
      # now evaluate the inverse
      # first need to find the additive constant for our integrated rate,
      # which comes from the evaluation of all the complete segments prior to
      # the one where our value of E_sample lies
      #constant = np.sum([x.inverse_int_domain_upper - x.inverse_int_domain_lower for x in self.hull_list[0:hull_segment_idx]])
      poisson_sample = self.hull_list[hull_segment_idx].evaluate_inverse(E_sample)#, constant)
      return poisson_sample
    except IndexError:
      return 0.0

  def eval_hull(self, time):
    # find the hull segment we should sample from
    less_than_upper = [time <= x.left for x in self.hull_list]
    # now find the first hull segment where our exponential variable is
    # greater than the lower bound of the range, as this will be the one
    # where we need to sample from inverse
    try:
      hull_segment_idx = np.where(less_than_upper)[0][-1]
    except IndexError:
      hull_segment_idx = 0
    # now evaluate the inverse
    return self.hull_list[hull_segment_idx].evaluate_hull(time)


  def eval_inverse_integrated(self, time=None, sample=100):
    if(time is None):
      time = np.linspace(self.hull_list[0].inverse_integrated_domain_lower,
                         self.hull_list[-1].inverse_integrated_domain_upper, 100)
    # for hull in self.hull_list:
    #   print(hull)
    #   # print('inverse domain = ( {}, {} )'.format(hull.inverse_integrated_domain_lower,
    #   #                                            hull.inverse_integrated_domain_upper))
    #   # print('constant = {}'.format(hull.constant))
    inverse = []
    for t in time:
      #print('t = {}, num_samples = {}'.format(t, len(inverse)))
      less_than_upper = [t <= x.inverse_integrated_domain_upper for x in self.hull_list]
      #print(less_than_upper)
      #print(np.where(less_than_upper)[0])
      hull_segment_idx = np.where(less_than_upper)[0][0]
      #constant = np.sum([x.inverse_integrated_range_upper - x.inverse_integrated_range_lower for x in self.hull_list[0:hull_segment_idx]])
      inverse.append(self.hull_list[hull_segment_idx].evaluate_inverse(t))#, constant))
    return time, np.array(inverse)


  def eval_integrated(self, time=None, sample=100):
    if(time is None):
      time = np.linspace(self.hull_list[0].integrated_domain_lower,
                         self.hull_list[-1].integrated_domain_upper, 100)
    integrated = []
    for t in time:
      #print('t = {}'.format(t))
      less_than_upper = [t <= x.integrated_domain_upper for x in self.hull_list]
      #print(less_than_upper)
      #print(np.where(less_than_upper)[0])
      hull_segment_idx = np.where(less_than_upper)[0][0]
      #constant = np.sum([x.inverse_integrated_domain_upper - x.inverse_integrated_domain_lower for x in self.hull_list[0:hull_segment_idx]])
      integrated.append(self.hull_list[hull_segment_idx].evaluate_integrated(t))#, constant))
    return time, np.array(integrated)



class HullNode(object):
  def __init__(self, m, b, left, right, pr=0.0):
    self.m = m
    self.b = b
    self.left = left
    self.right = right
    self.pr = pr
    # adding range for the method as it will help us out later
    self.range_lower = self.m * self.left + self.b
    self.range_upper = self.m * self.right + self.b
    self.integrated_domain_lower = 0.0 #self.left
    self.integrated_domain_upper = self.right
    # need to evaluate with any previous nodes to find the range
    self.integrated_range_lower = 0.0
    self.integrated_range_upper = 0.0
    # similarly, need to evaluate with any previous nodes to find the domain of
    # the inverse of the integral, which will be the same as the range of the
    # integrated rate
    self.inverse_integrated_domain_lower = 0.0
    self.inverse_integrated_domain_upper = 0.0
    self.inverse_integrated_range_lower = 0.0 #self.left
    self.inverse_integrated_range_upper = self.right
    # add a constant term, which will be found when evaluating the integral
    # of all the hull sections
    self.constant = 0.0
    self.epsilon = 0.0000001

  def evaluate_inverse(self, sample):#, constant):
    """ will evaluate the inverse rate of this segment at `sample`"""
    if(np.abs(self.m) > self.epsilon):
       return (-self.b + np.sqrt(np.square(self.b) - 2.0 * self.m * (self.constant - sample))) / self.m
    else:
       return sample / self.b

  def evaluate_integrated(self, sample):#, constant=0.0):
    """ will evaluate the inverse rate of this segment at `sample`"""
    if(np.abs(self.m) > self.epsilon):
       return self.m * sample ** 2.0 / 2.0 + self.b * sample + self.constant
    else:
       return self.b * sample

  def evaluate_hull(self, sample):
    return self.m * sample + self.b

  def __eq__(self, other):
    from math import isclose

    def close(a, b):
      if a is b is None:
        return True
      if (a is None and b is not None) or (b is None and a is not None):
        return False
      return isclose(a, b, abs_tol=1e-02)

    return all((
      close(self.m, other.m), close(self.left, other.left),
      close(self.right, other.right),
      close(self.pr, other.pr)
    ))

  def __repr__(self):
    return "HullNode(m={m}, b={b}, left={left}, right={right}, pr={pr})".format(
      m=self.m, b=self.b, left=self.left, right=self.right, pr=self.pr
    )

  def __hash__(self):
    return hash(str(self))


def compute_hulls(S, fS, domain):
  """
  (Re-)compute upper and lower hull given
  the segment points `S` with function values
  `fS` and the `domain` of the logpdf.

  Parameters
  ----------
  S : np.ndarray (N, 1)
     Straight-line segment points accumulated thus far.

  fS : tuple
    Value of the `logpdf` under sampling for each
    of the given segment points in `S`.

  domain : Tuple[float, float]
    Domain of `logpdf`.
    May be unbounded on either or both sides,
    in which case `(float("-inf"), float("inf"))`
    would be passed.
    If this domain is unbounded to the left,
    the derivative of the logpdf
    for x<= a must be positive.
    If this domain is unbounded to the right          the derivative of the logpdf for x>=b
    must be negative.

  Returns
  ----------
  lower_hull: List[arspy.hull.HullNode]
  upper_hull: List[arspy.hull.HullNode]

  """
  assert(len(S) == len(fS))
  assert(len(domain) == 2)

  lower_hull = []

  for li in range(len(S) - 1):
    m = (fS[li + 1] - fS[li]) / (S[li + 1] - S[li])
    b = fS[li] - m * S[li]
    left = S[li]
    right = S[li + 1]
    lower_hull.append(HullNode(m=m, b=b, left=left, right=right))

  # compute upper piecewise-linear hull

  # expected final length of upper hull after full computation
  n_upper_segments = 2 * (len(S) - 2) + isinf(domain[0]) + isinf(domain[1])

  upper_hull = []

  if isinf(domain[0]):
    # first line (from -infinity)

    m = (fS[1] - fS[0]) / (S[1] - S[0])
    b = fS[0] - m * S[0]
    pr = compute_segment_log_prob(float("-inf"), S[0], m, b)

    upper_hull.append(
      HullNode(m=m, b=b, pr=pr, left=float("-inf"), right=S[0])
    )

  # second line
  m = (fS[2] - fS[1]) / (S[2] - S[1])
  b = fS[1] - m * S[1]
  pr = compute_segment_log_prob(S[0], S[1], m, b)

  upper_hull.append(HullNode(m=m, b=b, pr=pr, left=S[0], right=S[1]))

  # interior lines
  # there are two lines between each abscissa
  for li in range(1, len(S) - 2):
    print('li = {}'.format(li))
    # for hull in upper_hull:
    #   print(hull)
    # print(upper_hull[-1])
    # print(upper_hull[-1].left)
    # print(upper_hull[-1].right)
    # print(S[li - 1])
    # if((upper_hull[-1].left == S[li - 1]) & (upper_hull[-1].right == S[li])):
    #   continue
    m1 = (fS[li] - fS[li - 1]) / (S[li] - S[li - 1])
    b1 = fS[li] - m1 * S[li]

    m2 = (fS[li + 2] - fS[li + 1]) / (S[li + 2] - S[li + 1])
    b2 = fS[li + 1] - m2 * S[li + 1]

    # if isinf(m1) and isinf(m2):
    #   raise ValueError("both hull slopes are infinite")

    dx1 = S[li] - S[li - 1]
    df1 = fS[li] - fS[li - 1]
    dx2 = S[li + 2] - S[li + 1]
    df2 = fS[li + 2] - fS[li + 1]

    f1 = fS[li]
    f2 = fS[li + 1]
    x1 = S[li]
    x2 = S[li + 1]

    # more numerically stable than above
    # this is where the lines intersect
    ix = ((f1 * dx1 - df1 * x1) * dx2 - (f2 * dx2 - df2 * x2) * dx1) / (df2 * dx1 - df1 * dx2)
    if isinf(m1) or abs(m1 - m2) < 10.0 ** 8 * eps(m1):
      ix = S[li]
      pr1 = float("-inf")
      pr2 = compute_segment_log_prob(ix, S[li + 1], m2, b2)
    elif isinf(m2):
      ix = S[li + 1]
      pr1 = compute_segment_log_prob(S[li], ix, m1, b1)
      pr2 = float("-inf")
    # else:
    #   if isinf(ix):
    #     raise ValueError("Non finite intersection")

    if abs(ix - S[li]) < 10.0 ** 12 * eps(S[li]):
      ix = S[li]
    elif abs(ix - S[li + 1]) < 10.0**12 * eps(S[li + 1]):
      ix = S[li + 1]
      ######################################
      # this is where I am changing the code
    if (ix < S[li]) or (ix > S[li + 1]) or isinf(ix) or isnan(ix) or (isinf(m1) and isinf(m2)):
      print('here')
      # only add on segment that goes from
      # (S[li], fS[li]) to (S[li + 1], fS[li + 1])
      m = (fS[li + 1] - fS[li]) / (S[li + 1] - S[li])
      b = fS[li + 1] - m * S[li + 1]
      # pr = compute_segment_log_prob(S[li], ix, m, b)
      pr = 0.0
      upper_hull.append(HullNode(m=m, b=b, pr=pr, left=S[li], right=S[li + 1]))

    else:
      # can fit this segment using normal ARS
      pr1 = compute_segment_log_prob(S[li], ix, m1, b1)
      pr2 = compute_segment_log_prob(ix, S[li + 1], m2, b2)


      print('ix = {}'.format(ix))
      print('S[li] = {}'.format(S[li]))
      print('S[li + 1] = {}\n'.format(S[li + 1]))
      upper_hull.append(HullNode(m=m1, b=b1, pr=pr1, left=S[li], right=ix))
      upper_hull.append(HullNode(m=m2, b=b2, pr=pr2, left=ix, right=S[li + 1]))

  # second last line
  m = (fS[-2] - fS[-3]) / float(S[-2] - S[-3])
  b = fS[-2] - m * S[-2]
  pr = compute_segment_log_prob(S[-2], S[-1], m, b)

  upper_hull.append(HullNode(m=m, b=b, pr=pr, left=S[-2], right=S[-1]))

  if isinf(domain[1]):
    # last line (to infinity)
    m = (fS[-1] - fS[-2]) / (S[-1] - S[-2])
    b = fS[-1] - m * S[-1]
    pr = compute_segment_log_prob(S[-1], float("inf"), m, b)

    upper_hull.append(
      HullNode(m=m, b=b, pr=pr, left=S[-1], right=float("inf"))
    )

  # normalize probabilities
  normalized_probabilities = exp_normalize(
    asarray([node.pr for node in upper_hull])
  )

  for node, probability in zip(upper_hull, normalized_probabilities):
    node.pr = probability

  #assert(len(lower_hull) == len(S) - 1)
  #assert(len(upper_hull) == n_upper_segments)

  return lower_hull, upper_hull


def compute_segment_log_prob(l, r, m, b):
  if l == float("-inf"):
    return -log(m) + m * r + b
  elif r == float("inf"):
    return -log(-m) + m * l + b

  M = max(m * r + b, m * l + b)

  return -log(abs(m)) + log(abs(exp(m * r + b - M) - exp(m * l + b - M))) + M


def sample_upper_hull(upper_hull, random_stream):
  """
  Return a single value randomly sampled from
  the given `upper_hull`.

  Parameters
  ----------
  upper_hull : List[pyars.hull.HullNode]
    Upper hull to evaluate.

  random_stream : numpy.random.RandomState
    (Seeded) stream of random values to use during sampling.

  Returns
  ----------
  sample : float
    Single value randomly sampled from `upper_hull`.

  """
  cdf = cumsum([node.pr for node in upper_hull])

  # randomly choose a line segment
  U = random_stream.rand()

  node = next(
    (node for node, cdf_value in zip(upper_hull, cdf) if U < cdf_value),
    upper_hull[-1]  # default is last line segment
  )

  # sample along that line segment
  U = random_stream.rand()

  m, left, right = node.m, node.left, node.right

  M = max(m * right, m * left)
  x = (log(U * (exp(m * right - M) - exp(m * left - M)) + exp(m * left - M)) + M) / m

  assert(x >= left and x <= right)

  if isinf(x) or isnan(x):
    raise ValueError("sampled an infinite or 'nan' x")

  return x


def evaluate_hulls(x, lower_hull, upper_hull):

  # lower bound
  lh_val = 0.0

  if x < min(node.left for node in lower_hull):
    lh_val = float("-inf")
  elif x > max(node.right for node in lower_hull):
    lh_val = float("-inf")
  else:
    node = next(node for node in lower_hull if node.left <= x <= node.right)
    lh_val = node.m * x + node.b

  # upper bound
  uh_val = 0.0

  node = next(node for node in upper_hull if node.left <= x <= node.right)
  uh_val = node.m * x + node.b

  return lh_val, uh_val
