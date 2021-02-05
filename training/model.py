import tensorflow as tf
import numpy as np

#from training.layers import *

def Generator(x):
    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    # project and reshape
    x = tf.keras.layers.Dense(units=4*4*1024, use_bias=False, kernel_initializer=w_init, name='project')(x)
    x = tf.keras.layers.Reshape((4, 4, 1024))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)    

    #conv1
    x = tf.keras.layers.Conv2DTranspose(512, kernel_size=5, strides=(2,2), padding='SAME', kernel_initializer=w_init, name='Conv1')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)    

    #oncv2
    x = tf.keras.layers.Conv2DTranspose(256, kernel_size=5, strides=(2,2), padding='SAME', kernel_initializer=w_init, name='Conv2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    #oncv3
    x = tf.keras.layers.Conv2DTranspose(128, kernel_size=5, strides=(2,2), padding='SAME', kernel_initializer=w_init, name='Conv3')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    #conv4
    x = tf.keras.layers.Conv2DTranspose(3, kernel_size=5, strides=(2,2), padding='SAME', kernel_initializer=w_init, name='Conv4')(x)
    imgs = tf.keras.activations.tanh(x)

    return imgs

def Discriminator(x):
    w_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    #conv1
    x = tf.keras.layers.Conv2D(128, kernel_size=5, strides=(2,2), padding='SAME', kernel_initializer=w_init, name='Conv1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    #conv2
    x = tf.keras.layers.Conv2D(256, kernel_size=5, strides=(2,2), padding='SAME', kernel_initializer=w_init, name='Conv2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    #conv3
    x = tf.keras.layers.Conv2D(512, kernel_size=5, strides=(2,2), padding='SAME', kernel_initializer=w_init, name='Conv3')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    #conv4
    x = tf.keras.layers.Conv2D(1024, kernel_size=5, strides=(2,2), padding='SAME', kernel_initializer=w_init, name='Conv4')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    #output
    
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(1, kernel_initializer=w_init)(x)
    
    return output
