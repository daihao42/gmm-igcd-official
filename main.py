#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/10/20
# @Author  : Hao Dai

import os
import sys
import time
import argparse

import numpy as np

import torch

import jax

from dataloaders.cifar100 import CIFAR100Loader
from dataloaders.tinyimagenet import TinyImageNetLoader
from dataloaders.imagenet100 import ImageNet100Loader

from classifier.pegmm import PEGMMClassifier
from classifier.mpegmm import MPEGMMClassifier
from classifier.mngmm import MNGMMClassifier
from classifier.mngmm_diag import MNGMMDiagClassifier

from clustering.kmeans import KMeansCluster
from clustering.kmeansmahalanobis import KMeansMahalanobisCluster
from clustering.gmm import GMMCluster

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# get the current time with a format yyyyMMdd-HHmm
def get_current_time():
    return time.strftime("%Y%m%d-%H%M", time.localtime())

# extract features
def extract_pretrained_features():
    pass

def Clustering_alg(alg):
    if alg == 'gmm':
        return GMMCluster
    elif alg == 'kmeans':
        return KMeansCluster
    elif alg == 'kmeansmahalanobis':
        return KMeansMahalanobisCluster
    else:
        raise ValueError('Clustering algorithm not supported')

def Classifier_alg(alg):
    if alg == 'pegmm':
        return PEGMMClassifier
    elif alg == 'mpegmm':
        return MPEGMMClassifier
    elif alg == 'mngmm':
        return MNGMMClassifier
    elif alg == 'mngmm_diag':
        return MNGMMDiagClassifier
    else:
        raise ValueError('Classifier algorithm not supported')


def Dataloader(args):
    # make data loader
    if args.dataset == 'cifar100':
        cifar100loader = CIFAR100Loader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = cifar100loader.makeHappyLoader()

    elif args.dataset == 'tinyimagenet':
        tinyimagenetloader = TinyImageNetLoader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = tinyimagenetloader.makeHappyLoader()

    elif args.dataset == 'imagenet100':
        imageNet100Loader = ImageNet100Loader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = imageNet100Loader.makeHappyLoader()
    
    else:
        raise ValueError('Dataset not supported')
        
    #train_loader, test_loader, test_old_loader, test_all_loader = makeMetaGCDLoader()
    #train_loader, test_loader, test_old_loader, test_all_loader = makeClassIncrementalLoader()

    #train_loader, test_loader, test_old_loader, test_all_loader = cifar100loader.makeMetaGCDLoader()
    #train_loader, test_loader, test_old_loader, test_all_loader = cifar100loader.makeClassIncrementalLoader()

    return train_loader, test_loader, test_old_loader, test_all_loader


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Generalized Class Incremental Learning')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset to learn')
    parser.add_argument('--data_dir', type=str, default='datasets/cifar100', help='Directory to the data')
    parser.add_argument('--pretrained_model_name', type=str, default='dino-vitb16', help='Name of the model')
    parser.add_argument('--base', type=int, default=50, help='Number of base classes')
    parser.add_argument('--increment', type=int, default=10, help='Number of incremental classes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--trail_name', type=str, default=f'', help='Name of the trail')
    parser.add_argument('--clustering-alg', type=str, default='gmm', help='Clustering algorithm')
    parser.add_argument('--classifier-alg', type=str, default='mngmm', help='Classifier algorithm')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes for the classifier')
    parser.add_argument('--num_dim', type=int, default=384, help='Number of features\' dim for the classifier')
    parser.add_argument('--with_early_stop', default=True, action=argparse.BooleanOptionalAction, help='Whether to use early stop')
    args = parser.parse_args()

    # Set the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng_key = jax.random.PRNGKey(args.seed)

    Clustering = Clustering_alg(args.clustering_alg)

    Classifier = Classifier_alg(args.classifier_alg)

    train_loader, test_loader, test_old_loader, test_all_loader = Dataloader(args)

    
    # if saved_models dir does not exist, create it
    log_saved_dir = f"{args.trail_name}_{get_current_time()}"
    if not os.path.exists(f"logs/{log_saved_dir}/saved_models"):
        os.makedirs(f"logs/{log_saved_dir}/saved_models")

    # same classifier for all stages, static expension
    s_classifier = Classifier(num_classes=args.num_classes, num_dim=args.num_dim, with_early_stop=args.with_early_stop)
    # for tinyimagenet
    #s_classifier = Classifier(num_classes=200, num_dim=384, num_samples=0, grid_bounds=(-10., 10.))

    s_classifier.init_parameters(n_epochs=1500, lr=4e-6, log_dir=f"logs/{log_saved_dir}/log/stage0", save_dir=f"logs/{log_saved_dir}/saved_models/stage0", batch_size=128)

    for i, (train_data, test_data, test_old_data, test_all_data) in enumerate(zip(train_loader, test_loader, test_old_loader, test_all_loader)): 
        if i == 0:
            testing_set = {'test_old': test_data, 'test_all': test_data, 'known_test': test_data}
            s_classifier.run(train_data._x, train_data._y, test_data._x, test_data._y, current_stage=i, testing_set=testing_set)
            known_test_data = test_data
        else:

            label_offset = args.base + (i-1)*args.increment

            clustering = Clustering(num_classes=args.increment, label_offset=args.base + (i-1)*args.increment)

            print(args.increment, (i-1)*args.increment)
            clustering.fit(train_data._x)
            pred = clustering.predict(train_data._x, train_data._y, with_known=True)
            print(np.unique(pred, return_counts=True))
            # output count of each class in train_data._y

            # for ngmm merge
            s_classifier._set_label_offset(label_offset)

            s_classifier.update_dir_infos(log_dir=f"logs/{log_saved_dir}/log/stage{i}", save_dir=f"logs/{log_saved_dir}/saved_models/stage{i}")

            testing_set = {'test_old': test_old_data, 'test_all': test_all_data, 'known_test': known_test_data}

            s_classifier.run(train_data._x, pred, test_data._x, test_data._y, current_stage=i, testing_set=testing_set)
