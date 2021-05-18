'''
Copyright (c) 2020. IIP Lab, Wuhan University
Author: Yaochen Zhu
'''

import os
import sys

import numpy as np
import pandas as pd

from tensorflow import keras

sys.path.append("libs")
from utils import load_records

class VaeDataGenerator(keras.utils.Sequence):
    '''
        Generate the training and validation
        data for the vae part of cvae model.
    '''
    def __init__(self,
                 content_path,
                 batch_size,
                 indexes,
                 pre_layers,
                 noise_type=None,
                 shuffle=True):
    
        self.contents = np.load(content_path)[indexes]

        '''
            In greedy layer wise pretraining, you need to 
            fixed the weights from the previous layers and 
            train the latter ones.
        '''
        for pre_layer in pre_layers:
            self.contents = pre_layer(self.contents)

        ### Basic info. of the dataset
        self.num_items = len(self.contents)
        self.item_idxes = np.arange(self.num_items)
        self.batch_size = batch_size

        ### Whether or not, or add which type of noise.
        self.noise_type = noise_type

        ### Shuffle the items if necessary.
        self.shuffle = shuffle 
        self.on_epoch_end()

    def on_epoch_end(self):
        '''
            Shuffle the item index after each epoch.
        '''
        if self.shuffle:
            np.random.shuffle(self.item_idxes)

    def __len__(self):
        '''
            The total number of batches.
        '''
        self.batch_num = self.num_items//self.batch_size + 1
        return self.batch_num

    def __getitem__(self, i):
        '''
            Return the batch indexed by i.
        '''
        batch_idxes = self.item_idxes[i*self.batch_size:(i+1)*self.batch_size]
        batch_target = self.contents[batch_idxes]
        if self.noise_type is None:
            batch_input = batch_target
        else:
            batch_input = self.add_noise(self.noise_type, batch_target)
        return batch_input, batch_target

    def add_noise(self, noise_type, contents):
        '''
            For the peripheral layer pairs, corrupt
            their inputs and train them as SDAE style.
        '''
        if noise_type == 'Gaussian':
            noise = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return contents + noise
        if 'Mask' in noise_type:
            frac = float(noise_type.split('-')[1])
            masked_contents = np.copy(contents)
            for item in masked_contents:
                zero_pos = np.random.choice(len(item), int(round(
                    frac*len(item))), replace=False)
                item[zero_pos] = 0
            return masked_contents

    @property
    def feature_dim(self):
        return self.contents.shape[-1]
    

class CVaeDataGenerator(keras.utils.Sequence):
    '''
        Generate the data for cvae model.
    '''
    def __init__(self, 
                 user_record_path,
                 item_record_path,        
                 content_path,
                 batch_size,
                 latent_size,
                 params,
                 shuffle):

        self.contents = np.load(content_path)   
        self.user_records = load_records(user_record_path)
        self.item_records = load_records(item_record_path)

        self.params = params
        self.latent_size = latent_size
        self.num_users = len(self.user_records)
        self.num_items = len(self.item_records)

        ### Initialize user, item embeddings
        self.U = np.random.randn(self.num_users, latent_size)*params.lambda_U
        self.V = np.random.randn(self.num_items, latent_size)*params.lambda_V
        self.Theta = np.random.randn(self.num_items, latent_size)*params.lambda_V

        ### Users and items that have at least one interactions 
        self.lucky_items = [len(items)>0 for items in self.item_records]
        self.active_users = [len(users)>0 for users in self.user_records]

        ### For training the vae
        self.item_idxes = np.arange(self.num_items)
        self.batch_size = batch_size

        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()

    def on_epoch_end(self):
        '''
            Shuffle the item index after each epoch.
        '''
        if self.shuffle:
            np.random.shuffle(self.item_idxes)

    def __len__(self):
        '''
            The total number of batches.
        '''
        self.batch_num = self.num_items // self.batch_size + 1
        return self.batch_num

    def __getitem__(self, i):
        '''
            Return the batch indexed by i.
        '''
        batch_idxes = self.item_idxes[i*self.batch_size:(i+1)*self.batch_size]
        batch_hidden = self.V[batch_idxes]
        batch_content = self.contents[batch_idxes]
        return [batch_content, batch_hidden], batch_content

    def update_U(self, U):
        assert U.shape == self.U.shape
        del self.U; self.U = U

    def update_V(self, V):
        assert V.shape == self.V.shape
        del self.V; self.V = V

    def update_Theta(self, Theta):
        assert Theta.shape == self.Theta.shape
        del self.Theta; self.Theta = Theta