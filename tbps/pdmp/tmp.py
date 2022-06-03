import tensorflow as tf
import sys

if __name__ == '__main__':

  @tf.function
  def body(i):
    b = tf.constant(1.0)
    i = i + b
    tf.print('i = {}'.format(i), output_stream=sys.stdout)
    return (i, )

  cond = lambda i: tf.less(i, 10)

  x = tf.constant(0.0)

  y = tf.while_loop(cond, body, loop_vars=[x])
  tf.print('y = {}'.format(y), output_stream=sys.stdout)
