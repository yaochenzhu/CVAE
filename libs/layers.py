'''
Copyright (c) 2020. IIP Lab, Wuhan University
Author: Yaochen Zhu
'''

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import backend as K

from tensorflow_probability import distributions as tfp


def binary_crossentropy(y_true, y_pred):
    '''
        The tensorflow style binary crossentropy
    '''
    loss = -tf.reduce_mean(
        tf.reduce_sum(
            y_true * tf.log(tf.maximum(y_pred, 1e-16)) + (1-y_true) * 
            tf.log(tf.maximum(1-y_pred, 1e-16)), 1
        ))
    return loss


def mse(y_true, y_pred):
    '''
        The tensorflow style mean squareed error
    '''
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), 1))
    return loss

class TransposedSharedDense(layers.Layer):
    '''
        Dense layer that shares weights (transposed) and bias 
        with another dense layer. Commonly used in symetric
        stucture of auto-encoders. To fetch weights from the 
        source dense layerccall its .weights attributes.
    '''

    def __init__(self, weights, activation, **kwargs):
        super(TransposedSharedDense, self).__init__(**kwargs)
        assert(len(weights) in [1, 2]), \
            "Specify the [kernel] or the [kernel] and [bias]."
        self.W = weights[0]

        if len(weights) == 1:
            b_shape = self.W.shape.as_list()[0]
            self.b = self.add_weight(shape=(b_shape),
                name="bias",
                trainable=True,
                initializer="zeros")
        else:
            self.b = weights[1]
        self.activate = activations.get(activation)

    def call(self, inputs):
        return self.activate(K.dot(inputs, K.transpose(self.W))+self.b)


class AddGaussianLoss(layers.Layer):
    '''
        Add the KL divergence between the variational 
        Gaussian distribution and the prior to loss.
    '''
    def __init__(self, 
                 **kwargs):
        super(AddGaussianLoss, self).__init__(**kwargs)             
        self.lamb_kl = self.add_weight(shape=(), 
                                       name="lamb_kl", 
                                       initializer="ones", 
                                       trainable=False)

    def call(self, inputs):
        mu, std  = inputs
        ### The one used by the tensorflow community
        kl_loss =  self.lamb_kl * 0.5 * tf.reduce_mean(
            tf.reduce_sum(tf.square(mu) + tf.square(std) - 2 * tf.log(std) - 1, 1
        ))

        ### Or you can use the tfp package
        '''
        var_dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        pri_dist = tfp.MultivariateNormalDiag(loc=K.zeros_like(mu), 
                                              scale_diag=K.ones_like(std))    
        kl_loss  = self.lamb_kl*K.mean(tfp.kl_divergence(var_dist, pri_dist))
        '''
        return kl_loss


class ReparameterizeGaussian(layers.Layer):
    '''
        Rearameterization trick for Gaussian
    '''
    def __init__(self, **kwargs):
        super(ReparameterizeGaussian, self).__init__(**kwargs)

    def call(self, stats):
        mu, std = stats
        dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        return dist.sample()