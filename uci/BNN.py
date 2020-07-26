from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal




 # DATA
X_train = np.loadtxt('x_data.txt', delimiter=",")
y_train = np.loadtxt('y_data.txt', delimiter=",")
X_train = X_train.reshape((80, 1))
y_train = y_train.reshape((80, ))

def main(_):
  def neural_network(X):
    h = tf.tanh(tf.matmul(X, W_0) + b_0)
    h = tf.matmul(h, W_1) + b_1
    return tf.reshape(h, [-1])
  ed.set_seed(42)


  # MODEL
  with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([1, 32]), scale=tf.ones([1, 32]),
                 name="W_0")
    W_1 = Normal(loc=tf.zeros([32, 1]), scale=tf.ones([32, 1]), name="W_1")
    b_0 = Normal(loc=tf.zeros(32), scale=tf.ones(32), name="b_0")
    b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_1")


    X = tf.placeholder(tf.float32, [80, 1], name="X")
    y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(80), name="y")

  # INFERENCE
  with tf.variable_scope("posterior"):
    with tf.variable_scope("qW_0"):
      loc = tf.get_variable("loc", [1, 32])
      scale = tf.nn.softplus(tf.get_variable("scale", [1, 32]))
      qW_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qW_1"):
      loc = tf.get_variable("loc", [32, 1])
      scale = tf.nn.softplus(tf.get_variable("scale", [32, 1]))
      qW_1 = Normal(loc=loc, scale=scale)

    with tf.variable_scope("qb_0"):
      loc = tf.get_variable("loc", [32])
      scale = tf.nn.softplus(tf.get_variable("scale", [32]))
      qb_0 = Normal(loc=loc, scale=scale)
    with tf.variable_scope("qb_1"):
      loc = tf.get_variable("loc", [1])
      scale = tf.nn.softplus(tf.get_variable("scale", [1]))
      qb_1 = Normal(loc=loc, scale=scale)

  inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                       W_1: qW_1, b_1: qb_1}, data={X: X_train, y: y_train})
  inference.run(logdir='log', n_samples=5, n_iter=10000)
  y_post = ed.copy(y, ({W_0: qW_0, b_0: qb_0,
                        W_1: qW_1, b_1: qb_1}))

  print("Mean squared error:")
  print(ed.evaluate('mean_squared_error', data={X: X_train, y_post: y_train}))

  print("Mean absolute error:")
  print(ed.evaluate('mean_absolute_error', data={X: X_train, y_post: y_train}))

if __name__ == "__main__":
  tf.app.run()