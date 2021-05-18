'''
Copyright (c) 2020. IIP Lab, Wuhan University
Author: Yaochen Zhu
'''

import os
import sys
sys.path.append("libs")

import numpy as np
import scipy as sp
import scipy.io as sio
from tensorflow.keras import optimizers

from layers import mse

class CollaborativeVAE():
    def  __init__(self,
                  cvae_datagen,
                  cvae_loss,
                  learning_rate,
                  pretrained_vae):

        self.data_gen = cvae_datagen
        self.params = self.data_gen.params

        self.vae = pretrained_vae.get_cvae_model(self.params.lambda_V, self.params.lambda_W)
        self.vae.compile(loss=cvae_loss, optimizer=optimizers.Adam(lr=learning_rate), metrics=[cvae_loss])

        self.encoder = pretrained_vae.get_inference_model()        
        self.num_users = self.data_gen.num_users
        self.num_items = self.data_gen.num_items
        self.latent_size = self.encoder.output.shape.as_list()[-1]

    def _precompute_U(self):
        '''
            Precompute the dominant part of A to reduce the 
            computational cost when optimizing the user embeddings
        '''
        V = self.data_gen.V
        lucky_items = self.data_gen.lucky_items            
        VVT = np.matmul(V[lucky_items].T, V[lucky_items])
        XX_U = VVT*self.params.B + np.eye(self.latent_size)*self.params.lambda_U
        return XX_U

    def _precompute_V(self):
        '''
            Precompute the dominant part of A to reduce the 
            computational cost when optimizing the item embeddings
        '''
        U = self.data_gen.U
        active_users = self.data_gen.active_users
        UUT = np.matmul(U[active_users].T, U[active_users])
        XX_V = UUT*self.params.B + np.eye(self.latent_size)*self.params.lambda_V
        return XX_V

    def update_vae(self):
        '''
            Update the vae model with new collaborative information
        '''
        self.vae.fit_generator(self.data_gen, epochs=1)
        new_Theta = self.encoder.predict_on_batch(self.data_gen.contents)
        self.data_gen.update_Theta(new_Theta)

    def update_embeddings(self):
        '''
            Update the user, item embeddings iteratively like the 
            probablistic matrix factorization algorithm with new 
            content information from VAE
        '''
        A_neg_B = self.params.A - self.params.B
        neg_loglld = np.exp(20) # Negative loglikelihood
        neg_loglld_old = 0.0 # neg_loglld from previous loop
        n_iter = 0

        while (n_iter<self.params.min_iter or (n_iter<self.params.max_iter and rel_improve>1e-6)):
            neg_loglld_old = neg_loglld
            neg_loglld = 0

            ### Update the user embeddings
            XX_U = self._precompute_U()
            cur_V = self.data_gen.V
            new_U = np.empty((self.num_users, self.latent_size), dtype=np.float32)
            for uid, liked_items in enumerate(self.data_gen.user_records):
                ### Every user rates at least some items.
                if len(liked_items) > 0:
                    A_U = XX_U + np.matmul(cur_V[liked_items].T, cur_V[liked_items])*A_neg_B
                    x_U = np.sum(cur_V[liked_items], axis=0)*self.params.A
                    new_U[uid] = sp.linalg.solve(A_U, x_U)
            self.data_gen.update_U(new_U)
            neg_loglld += 0.5*self.params.lambda_U*np.sum(new_U*new_U)

            ### Update the item embeddings
            XX_V = self._precompute_V()
            cur_U = self.data_gen.U
            cur_Theta = self.data_gen.Theta
            new_V = np.empty((self.num_items, self.latent_size), dtype=np.float32)
            for vid, visited_users in enumerate(self.data_gen.item_records):
                A_V = XX_V + np.matmul(cur_U[visited_users].T, cur_U[visited_users])*A_neg_B
                x_V = cur_Theta[vid]*self.params.lambda_V
                if len(visited_users) > 0:
                    x_V += np.sum(cur_U[visited_users], axis=0)*self.params.A
                    new_V[vid] = sp.linalg.solve(A_V, x_V)
                    B_V = A_V - np.eye(self.latent_size)*self.params.lambda_V
                    neg_loglld += 0.5*len(visited_users)*self.params.A
                    neg_loglld += 0.5*new_V[vid].dot(B_V).dot(new_V[vid][:,np.newaxis])
                    neg_loglld -= self.params.A*np.sum(np.dot(cur_U[visited_users], new_V[vid][:, np.newaxis]),axis=0)
                else:
                    new_V[vid] = sp.linalg.solve(A_V, x_V)
      
            self.data_gen.update_V(new_V)
            eps = new_V - cur_Theta
            neg_loglld += 0.5*self.params.lambda_V*np.sum(eps*eps)   
            
            ### Calculate the relative improvements
            rel_improve = abs(1.0*(neg_loglld_old - neg_loglld)/neg_loglld_old)
            n_iter = n_iter + 1

            print("#iteration: {}".format(n_iter))
            print("old neg_loglld: {}".format(neg_loglld_old))
            print("cur neg_loglld: {}".format(neg_loglld))
            print("relative improvements: {}".format(rel_improve))

        return neg_loglld

    def save_embeddings(self, save_root, save_mat=False):
        '''
            save the trained embeddings
        '''
        np.save(os.path.join(save_root, "U.npy"), self.data_gen.U)
        np.save(os.path.join(save_root, "V.npy"), self.data_gen.V)
        np.save(os.path.join(save_root, "Theta.npy"), self.data_gen.Theta)
        if save_mat:
            sio.savemat(os.path.join(save_root, "cvae.mat"), 
                {"m_U":self.data_gen.U, "m_V":self.data_gen.V, "m_theta":self.data_gen.Theta})