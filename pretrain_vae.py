'''
Copyright (c) 2020. IIP Lab, Wuhan University
Author: Yaochen Zhu
'''

import os
import time
import logging
import argparse

import sys
sys.path.append("libs")
from utils import Init_logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from data import VaeDataGenerator
from vae import LayerwisePretrainableVAE
from layers import binary_crossentropy

import warnings
warnings.filterwarnings('ignore')

### Fix the random seeds.
np.random.seed(0)
tf.set_random_seed(0)

hidden_sizes = [200, 100]
activations = ["sigmoid", "sigmoid"]
latent_size = 50
latent_activation = "sigmoid"

def get_lp_vae(feature_dim):
    lp_vae = LayerwisePretrainableVAE(
        input_shapes = [None, feature_dim],
        hidden_sizes = hidden_sizes,
        activations  = activations,
        latent_size  = latent_size,
        latent_activation = latent_activation
    )
    return lp_vae

if __name__ == '__main__':
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128,
        help="specify the batch size for pretraining")
    parser.add_argument("--device" , type=str, default="0",
        help="use which GPU device")
    parser.add_argument("--dataset", type=str, choices=["citeulike-a"],
        help="use which dataset for experiment")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Initialize the logging.
    Init_logging()

    ### Get the initial train, val data generator.
    content_file = os.path.join("data", args.dataset, "mult_nor.npy")
    total_num = np.load(content_file).shape[0]
    train_idxes = np.random.rand(total_num) < 0.8
    val_idxes = np.logical_not(train_idxes)

    pre_layers = []
    train_gen = VaeDataGenerator(
        content_path = content_file,
        batch_size = args.batch_size,
        pre_layers = pre_layers,
        indexes = train_idxes,
        noise_type = "Mask-0.3", shuffle = True
    )
    val_gen = VaeDataGenerator(
        content_path = content_file,
        batch_size = args.batch_size,
        pre_layers = pre_layers,
        indexes = val_idxes,           
        noise_type = "Mask-0.3", shuffle = True
    )

    ### Get the layerwise pretrainable vae model.
    lp_vae = get_lp_vae(train_gen.feature_dim)

    ### Pretrain the peripheral layer pairs.
    for i in range(len(hidden_sizes)):
        logging.info("Pretrain the {}th peripheral layer pair!".format(i+1))
        pretrain_peri = lp_vae.get_peri_for_pretraining(i)
        pretrain_peri.compile(optimizer=optimizers.Adam(lr=0.01),
                              loss=binary_crossentropy)
        pretrain_peri.fit_generator(train_gen, workers=4, 
                                    epochs=50, validation_data=val_gen)

        pre_layer = lp_vae.encoder.dense_list[i]
        pre_layers.append(K.function(pre_layer.input, pre_layer.output))

        del train_gen, val_gen

        '''
            Get the new data generator where the inputs are 
            processed by previous pretrained layers.
        '''
        train_gen = VaeDataGenerator(
            content_path = content_file,
            batch_size = args.batch_size,
            pre_layers = pre_layers,
            indexes = train_idxes, shuffle = True
        )
        val_gen = VaeDataGenerator(
            content_path = content_file,
            batch_size = args.batch_size,
            pre_layers = pre_layers,
            indexes = val_idxes, shuffle = True
        )

    ### Pretrain the latent core layer.
    logging.info("Pretrain the latent core layer!")
    core_epochs = 50
    pretrain_core = lp_vae.get_core_for_pretraining()
    pretrain_core.compile(optimizer=optimizers.Adam(lr=0.01), 
                          loss=binary_crossentropy)    
    pretrain_core.fit_generator(train_gen, workers=4, epochs=core_epochs, 
                                validation_data=val_gen)

    ### Pretrain the whole vae model.
    logging.info("Pretrain the whole vae model!")
    pre_layers = []
    train_gen = VaeDataGenerator(
        content_path = content_file,
        batch_size = args.batch_size,
        pre_layers = pre_layers,
        indexes  = train_idxes, shuffle = True
    )
    val_gen = VaeDataGenerator(
        content_path = content_file,
        batch_size = args.batch_size,
        pre_layers = pre_layers,
        indexes  = val_idxes, shuffle = True
    )
    epochs = 100
    vae_model = lp_vae.get_vae_model()
    vae_model.compile(optimizer=optimizers.Adam(lr=0.01), loss=binary_crossentropy, 
                      metrics=[binary_crossentropy])
    vae_model.fit_generator(train_gen, workers=4, epochs=epochs, 
                            validation_data=val_gen)

    ### Save the pretrained weights.
    save_root = os.path.join("models", args.dataset, "pretrained")
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    vae_model.save_weights(os.path.join(save_root, "weights.h5"))