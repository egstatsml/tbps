  def simulate_bounce_time_thinning(self, state, velocity, time):
    """
    """
    # find the current upper bound
    print('state befor fn = {}'.format(state))
    print('here exp')
    exp_d = tfd.Exponential(1.0)
    uni_d = tfd.Uniform()
    accepted = False
    proposed_time = 0.0
    while not accepted:
      _, grads = mcmc_util.maybe_call_fn_and_grads(
        self.target_log_prob_fn, state, name='sbps_simulate_bounce_time')
      upper_bound_rate = self.compute_upper_bound(grads, velocity)
      # now propose time with this rate
      print('proposed time')
      proposed_time += exp_d.sample(1) / upper_bound_rate
      print('proposed time = {}'.format(proposed_time))
      # now get the gradient at this time
      proposed_state = [s + v * proposed_time for s, v in zip(state, velocity)]
      _, grads = mcmc_util.maybe_call_fn_and_grads(
        self.target_log_prob_fn, proposed_state, name='sbps_simulate_bounce_time')
      # get the pointwise rate for this component
      print('getting rate')
      rate = self.compute_dot_grad_velocity(grads, velocity)
      print('rate = {}'.format(rate))
      proposal_u = uni_d.sample(1)
      #new_upper_bound = self.compute_upper_bound(grads, velocity)
      print('orig upper bound = {}'.format(upper_bound_rate))
      #print('new upper bound = {}'.format(new_upper_bound))
      ratio = rate / upper_bound_rate
      accepted_lambda = lambda: True
      rejected_lambda = lambda: False
      #time = proposed_time
      print('proposal_u = {}'.format(proposal_u))
      print('ratio = {}'.format(ratio))
      accepted = tf.case(
        [(tf.math.less(proposal_u, ratio), accepted_lambda)],
        default=rejected_lambda)
    print('state after fn = {}'.format(state))
    if proposed_time <= 0:
      return 10e6
    else:
      print('proposed_time = {}'.format(proposed_time[0]))
      return proposed_time[0]


  def compute_upper_bound(self, grads_target_log_prob, velocity):
    """ Compute upper bound using Cauchy-Schwarz inequality [1]

    Args:
      grads_target_log_prob (list(array)):
        List of arrays for the gradient of each variable.
      velocity (list(array)):
        List of arrays for the velocity of each variable.

     Returns:
       upper bound

    #### References:
    [1] https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality
    """
    # need to find the L2 Norm of all the state elements
    upper = tf.sqrt(self.compute_grad_l2_norm(grads_target_log_prob))
    # since the norm of the velocity will be one (current assumption)\
    # don't need to worry about computing it, just return the norm for
    # state parts
    return upper
