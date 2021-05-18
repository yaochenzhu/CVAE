'''
Copyright (c) 2020. IIP Lab, Wuhan University
Author: Yaochen Zhu
'''

import os
import sys
import argparse
sys.path.append("libs")

from utils import load_records

import numpy as np
import pandas as pd

M_list = list(range(50, 350, 50))

#python predict.py --dataset citeulike-a --embedding_dir models\citeulike-a\pvalue_1\embeddings\epoch_100
#python predict.py --dataset citeulike-a --embedding_dir models\citeulike-a\pvalue_10\embeddings\epoch_100

def recommend_item_to_users():
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["citeulike-a"],
        help="use which dataset for experiment")
    parser.add_argument("--p_value", type=int, default=10,
        help="num of interactions available during the training process")
    parser.add_argument("--embedding_dir", type=str, default=None,
        help="path where the trained user, item embeddings are stored")
    args = parser.parse_args()

    ### Load trained embeddings and user records, and calculate the scores
    U = np.load(os.path.join(args.embedding_dir, "U.npy"))
    V = np.load(os.path.join(args.embedding_dir, "V.npy"))
    scores = np.matmul(U, V.T)

    train_records = load_records(os.path.join("data", args.dataset, "cf-train-{}-users.dat".format(args.p_value)))
    test_records  = load_records(os.path.join("data", args.dataset, "cf-test-{}-users.dat".format(args.p_value)))

    train_idxes = np.zeros(scores.shape, dtype=np.float32)
    test_idxes  = np.zeros(scores.shape, dtype=np.float32)

    for i, (train_vids, test_vids) in enumerate(zip(train_records, test_records)):
        train_idxes[i, train_vids] = 1
        test_idxes [i, test_vids]  = 1

    ### Dismiss the training samples
    scores -= train_idxes*(scores.max(axis=-1)-scores.min(axis=-1))[:, np.newaxis]

    ### Get the item id sorted by the scores, and calculate the hits
    recomm_vids = np.argsort(-scores, axis=-1)
    hits = np.zeros(scores.shape, dtype=np.float32)
    for uid in range(len(recomm_vids)):
        hits[uid] = test_idxes[uid, recomm_vids[uid]]

    ### Evaluate the performance
    valid_uids = [uid for uid, vids in enumerate(test_records) if vids!=[]]    
    total_num = np.sum(test_idxes[valid_uids], axis=-1)
    recall_list = [[np.mean(np.sum(hits[valid_uids][:,:M], axis=-1) / total_num)]
        for M in M_list]
    print(recall_list)

    ### Save the evaluations
    recall_table = pd.DataFrame(dict(zip(M_list, recall_list)))
    save_path = os.path.join(args.embedding_dir, "recalls.txt")
    print("Evaluation results saved to: {}".format(save_path))
    recall_table.to_csv(save_path, index=False, sep=" ")

if __name__ == '__main__':
    recommend_item_to_users()