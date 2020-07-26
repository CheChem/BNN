import tensorflow as tf
reg_penalty = tf.losses.get_regularization_losses()
print(reg_penalty)