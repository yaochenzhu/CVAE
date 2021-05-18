'''
Copyright (c) 2020. IIP Lab, Wuhan University
Author: Yaochen Zhu
'''

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.engine import network

from layers import AddGaussianLoss, mse
from layers import ReparameterizeGaussian
from layers import TransposedSharedDense
from layers import ReparameterizeGaussian


class MLP(network.Network):
    '''
        The CVAE utilizes a stack of dense layers, 
        which is commonly known as the Multilayer 
        Perceptron (MLP), as its encoder.
    '''
    def __init__(self, 
                 hidden_sizes, 
                 activations,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.dense_list = []

        # Follow the original implementation,
        # add L2 regularizer only to the kernel.
        for i, (size, activation) in enumerate(zip(hidden_sizes, activations)):
            self.dense_list.append(
                layers.Dense(size, activation=activation,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    name="mlp_{}".format(i)
            ))

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        h_out = self.dense_list[0](x_in)
        for dense in self.dense_list[1:]:
            h_out = dense(h_out)
        self._init_graph_network(x_in, h_out)
        super(MLP, self).build(input_shapes)


class SymetricMLP(network.Network):
    '''
        The CVAE utilizes the symetric version of 
        its encoder, which reuses the transposed 
        weights and bias, as its decoder.
    '''
    def __init__(self,
                 source_mlp,
                 activations,
                 **kwargs):
        super(SymetricMLP, self).__init__(**kwargs)
        self.dense_list = []

        '''
            For the ith (i<L) layer of the decoder, use the kernel 
            and bias from the L-i+1th and L-ith layer of the encoder
        '''
        for i, (dense_W, dense_b) in enumerate(
            zip(source_mlp.dense_list[-1:0:-1], 
                source_mlp.dense_list[-2::-1])):
            ### Fetch the weights from source layers.
            weights = [dense_W.weights[0], dense_b.weights[1]]
            self.dense_list.append(
                TransposedSharedDense(weights=weights,
                activation=activations[i],
                name="symmetirc_mlp_{}".format(i)
            ))

        '''
            For the Lth layer of the decoder, it only uses the 
            tranposed weights from the 1th layer of the encoder, 
            and initializes the bias from scratch.
        '''
        weights = [source_mlp.dense_list[0].weights[0]]
        self.dense_list.append(
            TransposedSharedDense(weights=weights, activation=activations[-1]
        ))

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        h_out = self.dense_list[0](x_in)
        for dense in self.dense_list[1:]:
            h_out = dense(h_out)
        self._init_graph_network(x_in, h_out)
        super(SymetricMLP, self).build(input_shapes)


class Sampler(network.Network):
    '''
        Sample from the variational Gaussian, and add its KL 
        with the prior to loss
    '''
    def __init__(self, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.rep_gauss = ReparameterizeGaussian()

    def build(self, input_shapes):
        mean = layers.Input(input_shapes[0][1:])
        std  = layers.Input(input_shapes[1][1:])
        sample  = self.rep_gauss([mean, std])
        self._init_graph_network([mean, std], [sample])
        super(Sampler, self).build(input_shapes)


class LatentCore(network.Network):
    '''
        The latent core of CVAE, which takes the input, gets the
        mean and std of the hidden Gaussian, take a sample, and a
        pplies the first dense layer of the decoder to it.
    '''
    def __init__(self,  
                 latent_size, 
                 out_size, 
                 activation, 
                 **kwargs):
        super(LatentCore, self).__init__(**kwargs)
        self.dense_mean = layers.Dense(latent_size, name="mean")
        self.dense_std  = layers.Dense(latent_size, name="logstd")
        self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2), name="clip")
        self.exp = layers.Lambda(lambda x:K.exp(x), name="exp")
        self.sampler = Sampler(name="sampler")
        self.dense_out = layers.Dense(out_size, activation=activation, name="latent_out")

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        mean = self.dense_mean(x_in)
        std  = self.exp(self.clip(self.dense_std(x_in)))
        h_mid = self.sampler([mean, std]) 
        y_out = self.dense_out(h_mid)
        self._init_graph_network(x_in, [y_out, mean, std, h_mid])
        super(LatentCore, self).build(input_shapes)


class LayerwisePretrainableVAE():
    '''
        Finally, the VAE part of CVAE. For it to be layer-
        wise pretrainable, we seperately define the MLP 
        encoder, the latent core, and the decoder (which is 
        a symetric version of the encoder), in a laborous 
        manner. Now, its high time to put them together....
    '''
    def __init__(self, 
                 input_shapes,
                 hidden_sizes,
                 activations,
                 latent_size,
                 latent_activation):
        self.input_shapes = input_shapes
        self.latent_size = latent_size

        self.encoder = MLP(hidden_sizes, activations, name="encoder")
        self.encoder.build(input_shapes=input_shapes)
        self.latent_core = LatentCore(latent_size, hidden_sizes[-1], 
                                      latent_activation, name="latent_core")
        self.decoder = SymetricMLP(self.encoder, activations[::-1], name="decoder")
        
        ### Initialize the weights for latent core and decoder (Optitional)
        self.latent_core.build(input_shapes=[None, hidden_sizes[-1]])
        self.decoder.build(input_shapes=[None, hidden_sizes[-1]])

    def get_peri_for_pretraining(self, index, noise_level=None):
        '''
            Pair the ith layer of encoder and the L-i+1th 
            layer of decoder as an auto-encoder for pretraining
        '''
        depth = len(self.encoder.dense_list)
        assert index < depth
        src_dense = self.encoder.dense_list[index]
        sym_dense = self.decoder.dense_list[depth-index-1]
        x_in = layers.Input(shape=(src_dense.input.shape.as_list()[-1],),
                            name="pretrain_peri_{}_input".format(index))
        if noise_level is not None:
            gauss_mask = layers.GaussianNoise(stddev=noise_level, name="mask")
            x_nsy = gauss_mask(x_in)
            x_rec = sym_dense(src_dense(x_nsy))
        else:
            x_rec = sym_dense(src_dense(x_in))
        return models.Model(inputs=x_in, outputs=x_rec)

    def get_core_for_pretraining(self):
        '''
            Get the latent core for pretraining
        '''
        x_in = layers.Input(shape=(self.latent_core.input.shape.as_list()[-1],),
                            name="pretrain_core_input")
        x_rec, mean, std, _ = self.latent_core(x_in)
        latent_core_model = models.Model(inputs=x_in, outputs=x_rec)
        latent_core_model.add_loss(AddGaussianLoss()([mean, std]))
        return latent_core_model

    def get_vae_model(self):
        '''
            Get the whole vae model
        '''
        x_in = layers.Input(shape=self.input_shapes[1:], name="vae_model_input")
        h_mid = self.encoder(x_in)
        h_mid, mean, std, _ = self.latent_core(h_mid)
        x_rec = self.decoder(h_mid)
        vae_model = models.Model(inputs=x_in, outputs=x_rec)
        kl_loss = AddGaussianLoss()([mean, std])
        vae_model.add_loss(kl_loss)
        return vae_model

    def get_cvae_model(self, lambda_V, lambda_W):
        '''
            Get the vae part for cvae model
        '''
        x_in = layers.Input(shape=self.input_shapes[1:], name="cvae_content_input")
        V_in = layers.Input(shape=[self.latent_size,], name="cvae_embedding_input")
        h_mid = self.encoder(x_in)
        h_mid, mean, std, sample = self.latent_core(h_mid)
        x_rec = self.decoder(h_mid)
        cvae_model = models.Model(inputs=[x_in, V_in], outputs=x_rec)
        kl_loss = AddGaussianLoss()([mean, std])
        
        '''
            First, add extra mse between V and sample to the loss
        '''
        mse_loss = mse(V_in, sample)
        
        '''
            Then, add the L2 regularizer to the kernel of the first 
            two dense layers as MaP estimate for Gaussian variables.
        '''
        reg_loss = tf.nn.l2_loss(cvae_model.layers[1].layers[1].weights[0]) + \
                   tf.nn.l2_loss(cvae_model.layers[1].layers[2].weights[0]) + \
                   tf.nn.l2_loss(tf.transpose(cvae_model.layers[1].layers[1].weights[0])) + \
                   tf.nn.l2_loss(tf.transpose(cvae_model.layers[1].layers[2].weights[0]))        
        cvae_model.add_loss(kl_loss + lambda_V*mse_loss + lambda_W*reg_loss)
        return cvae_model


    def get_inference_model(self, return_std=False):
        '''
            Get the inference part of the vae model
        '''
        x_in = layers.Input(shape=self.input_shapes[1:], name="inference_model_input")
        h_mid = self.encoder(x_in)
        _, mean, _, _ = self.latent_core(h_mid)
        return models.Model(inputs=x_in, outputs=mean)


    def load_weights(self, weight_path):
        '''
            Load weights from pretrained vae
        '''
        vae = self.get_vae_model()
        vae.load_weights(weight_path)


if __name__ == "__main__":
    lp_vae = LayerwisePretrainableVAE(
        input_shapes = [None, 8000],
        hidden_sizes = [200, 100],
        activations  = ["sigmoid", "sigmoid"],
        latent_size  = 50,
        latent_activation = "sigmoid"
    )
    vae.summary()