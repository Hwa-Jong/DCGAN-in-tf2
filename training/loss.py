import tensorflow as tf


def loss(y_true, y_pred):
    loss = tf.reduce_mean( tf.compat.v1.losses.sigmoid_cross_entropy(y_true, y_pred) )    
    return loss

