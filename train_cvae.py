import os
import sys
import logging
import argparse
sys.path.append("libs")

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from utils import Init_logging
from cvae import CollaborativeVAE
from data import CVaeDataGenerator
from pretrain_vae import get_lp_vae
from layers import binary_crossentropy

class Params():
    def __init__(self):
        self.A = 1.
        self.B = 0.01
        self.lambda_U = 0.1
        self.lambda_V = 10
        self.lambda_W = 2e-4
        self.max_iter = 1
        self.min_iter = 1


if __name__ == '__main__':
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["citeulike-a"],
        help="use which dataset for experiment")
    parser.add_argument("--p_value", type=int, default=1,
        help="num of interactions available during the training process")
    parser.add_argument("--pretrained_path", type=str, default=None,
        help="path where the weights pretrained vae are stored")
    parser.add_argument("--batch_size", type=int, default=128,
        help="batch size for updating vae in the EM like optimization")
    parser.add_argument("--epochs", type=int, default=100,
        help="epochs for the EM like optimization of cvae")
    parser.add_argument("--device" , type=str, default="0",
        help="use which GPU device")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Fix the random seeds.
    np.random.seed(0)
    tf.set_random_seed(0)

    ### Initialize the logging.
    Init_logging()

    ### Setup the pretrained vae model.
    content_path = "data/{}/mult_nor.npy".format(args.dataset)
    feature_dim = np.load(content_path).shape[-1]
    pretrained_vae = get_lp_vae(feature_dim)

    if args.pretrained_path is None:
    	pretrained_path = "models/{}/pretrained/weights.h5".format(args.dataset)
    else:
    	pretrained_path = args.pretrained_path
    pretrained_vae.load_weights(pretrained_path)

    ### Get the data generater.
    user_record_path = "data/{}/cf-train-{}-users.dat".format(args.dataset, args.p_value)
    item_record_path = "data/{}/cf-train-{}-items.dat".format(args.dataset, args.p_value)

    cvae_datagen = CVaeDataGenerator(
        user_record_path = user_record_path,
        item_record_path = item_record_path,
        content_path = content_path,
        latent_size = pretrained_vae.latent_size,
        batch_size  = args.batch_size,
        params = Params(), shuffle = True
    )

    ### Set up the collaborative vae model.
    cvae_model = CollaborativeVAE(
        cvae_datagen = cvae_datagen,
        cvae_loss = binary_crossentropy,
        learning_rate = 0.001,        
        pretrained_vae = pretrained_vae
    )

    ### Use predict_on_batch or memory will leak.
    init_item_embeddings = cvae_model.encoder.predict_on_batch(cvae_datagen.contents)
    cvae_datagen.update_V(init_item_embeddings)
    cvae_datagen.update_Theta(init_item_embeddings)

    ### Save the pretrained weights.
    save_root = os.path.join("models", args.dataset, "pvalue_{}".format(args.p_value))
    embedding_fmt = os.path.join(save_root, "embeddings", "epoch_{}")
    weight_fmt = os.path.join(save_root, "cvae", "epoch_{}")

    ### Train the cvae model in an EM manner.
    for epoch in np.arange(args.epochs):
        logging.info("-"*10 + "Epoch: {}".format(epoch) + "-"*10)
        cvae_model.update_vae()
        neg_loglld = cvae_model.update_embeddings()
        logging.info("PMF step. Neg log-likelihood: {}".format(neg_loglld))

        if epoch >= 50 and (epoch+1) % 50 == 0:
            embedding_root = embedding_fmt.format(epoch+1)
            weight_root = weight_fmt.format(epoch+1)            
            if not os.path.exists(embedding_root):
                os.makedirs(embedding_root)
            if not os.path.exists(weight_root):
                os.makedirs(weight_root)
            cvae_model.save_embeddings(embedding_root, save_mat=True)
            cvae_model.vae.save_weights(os.path.join(weight_root, "weights.h5"))