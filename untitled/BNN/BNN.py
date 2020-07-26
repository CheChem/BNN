from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal




def main(_):
  def neural_network(X):
    h = tf.nn.sigmoid(tf.matmul(X, W_0) )
    h = tf.nn.sigmoid(tf.matmul(h, W_1) )
    h = tf.matmul(h, W_2)
    return tf.reshape(h, [-1])
  ed.set_seed(42)

  # DATA
  X_train = np.loadtxt('X1.txt', delimiter=",")
  y_train = np.loadtxt('Y1.txt', delimiter=",")
  X_train = X_train.reshape((50, 2))
  y_train = y_train.reshape((50, ))


  # MODEL
  with tf.name_scope("model"):
    W_0 = Normal(loc=tf.zeros([2, 2]), scale=tf.ones([2, 2]),
                 name="W_0")
    W_1 = Normal(loc=tf.zeros([2, 2]), scale=tf.ones([2, 2]), name="W_1")
    W_2 = Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1]), name="W_2")

    X = tf.placeholder(tf.float32, [50, 2], name="X")
    y = Normal(loc=neural_network(X), scale=0.1 * tf.ones(50), name="y")

  # INFERENCE
  with tf.variable_scope("posterior"):
    with tf.variable_scope("qW_0"):
      loc0 = tf.get_variable("loc", [2, 2])
      scale0 = tf.nn.softplus(tf.get_variable("scale", [2, 2]))
      qW_0 = Normal(loc=loc0, scale=scale0)
    with tf.variable_scope("qW_1"):
      loc1 = tf.get_variable("loc", [2, 2])
      scale1 = tf.nn.softplus(tf.get_variable("scale", [2, 2]))
      qW_1 = Normal(loc=loc1, scale=scale1)
    with tf.variable_scope("qW_2"):
      loc2 = tf.get_variable("loc", [2, 1])
      scale2 = tf.nn.softplus(tf.get_variable("scale", [2, 1]))
      qW_2 = Normal(loc=loc2, scale=scale2)

  inference = ed.KLqp({W_0: qW_0,
                       W_1: qW_1,
                       W_2: qW_2}, data={X: X_train, y: y_train})
  inference.run(n_samples=5, n_iter=10000)
  y_post = ed.copy(y, {W_0: qW_0,
                       W_1: qW_1,
                       W_2: qW_2})


  print(loc0,scale0)
  print(loc0.eval(), scale0.eval(),loc1.eval(), scale1.eval(),loc2.eval(), scale2.eval())
  print("Mean squared error on test data:")
  print(ed.evaluate('mean_squared_error', data={X: X_train, y_post: y_train}))

  print("Mean absolute error on test data:")
  print(ed.evaluate('mean_absolute_error', data={X: X_train, y_post: y_train}))

if __name__ == "__main__":
  tf.app.run()



