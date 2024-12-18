#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/10/20
# @Author  : Hao Dai
# Point Estimation Gaussian Mixture Model

import gpytorch

import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.tensorboard import SummaryWriter

from sklearn.decomposition import PCA
import joblib

from collections import defaultdict

from numpy import save
import pandas as pd
import numpy as np

from tqdm import tqdm

import copy

import math

import os
# to fix the bug 'UNKNOWN: Failed to determine best cudnn convolution algorithm'
# the reason is about overrate usage of GPU memory,
# jax would allocate 90% of GPU memory for application
# the following flags would set the allocation to be 70%
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="false"
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.35'

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp


# crreating classifier model
class MPEGMMClassifier():

    def __init__(self, num_samples, num_dim, num_classes, grid_bounds=(-10., 10.)):
        self.num_samples = num_samples
        self.num_dim = num_dim

        # Calculate means and covariance matrices for each class
        self.class_means = defaultdict()
        self.class_covariances = defaultdict()
        self.class_supports = defaultdict()

        self.pca = None

        self.global_params = None

        self.label_offset = 0


    def update_dir_infos(self, log_dir = "logs/", save_dir = "saved_models/"):
        self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir

        
    def init_parameters(self, n_epochs = 10, lr= 0.1, log_dir = "logs/", save_dir = "saved_models/", batch_size = 256,
                        train_likelihood_sample=8, test_likelihood_sample=16, use_cuda = True, num_samples = None):
        if num_samples is not None:
            self.num_samples = num_samples

        self._use_cuda = use_cuda

        self.batch_size = batch_size
        self.save_dir = save_dir

        self.writer = SummaryWriter(log_dir)


    def train(self, features, labels, current_epoch):

        features, labels = self.pre_processing(features, labels)
        # Get unique classes
        classes =  np.unique(labels).tolist()

        print('Training the model... on ', classes)

        last_params = {"class_means": copy.deepcopy(self.class_means), "class_covariances": copy.deepcopy(self.class_covariances)}

        # Calculate means and covariance matrices for each class
        for cls in tqdm(classes):
            class_features = features[labels == cls]
            class_means = np.mean(class_features, axis=0)
            class_covariances = np.cov(class_features, rowvar=False)
            class_supports = class_features.shape[0]
            self._merge_params(class_means, class_covariances, class_supports, cls)

        # Predict the classes of the training set
        predictions, class_probabilities = self._predict(features)

        self.class_means, self.class_covariances = last_params["class_means"], last_params["class_covariances"]

        # re-calculate the class means and covariances
        for cls in classes:
            class_features = features[predictions == cls]
            class_means = np.mean(class_features, axis=0)
            class_covariances = np.cov(class_features, rowvar=False)
            class_supports = class_features.shape[0]
            self._merge_params(class_means, class_covariances, class_supports, cls)

        predictions, class_probabilities = self._predict(features)

        correct = jnp.sum(predictions == labels).tolist()

        print('Train set: Accuracy: {}/{} ({}%)'.format(
            correct, len(features), 100. * correct / float(len(features))
        ))
        # log the test accuracy to tensorboard
        self.writer.add_scalar("Accuracy/train", 100. * correct / float(len(features)), current_epoch) 



    def _merge_params(self, class_means, class_covariances, class_supports, cls:int):

        if self.class_means.get(cls) is None:
            self.class_means[cls] = class_means
            self.class_covariances[cls] = class_covariances
            self.class_supports[cls] = class_supports
            return

        self.class_means[cls] = (class_means * class_supports + self.class_means[cls] * self.class_supports[cls]) / (class_supports + self.class_supports[cls])
        self.class_covariances[cls] = (class_covariances * class_supports + self.class_covariances[cls] * self.class_supports[cls]) / (class_supports + self.class_supports[cls])
        self.class_supports[cls] = class_supports + self.class_supports[cls]


    def _calculate_class_probabilities(self, samples, class_means, class_covariances):
        # Calculate class probabilities
        class_probabilities = defaultdict()

        for cls in class_means.keys():
            mean = class_means[cls]
            covariance = class_covariances[cls]

            # add a marginal value to the covariance matrix to avoid singular matrix
            covariance += jnp.eye(covariance.shape[0]) * 1e-6

            # Calculate the probability of each sample
            class_probability = jax.scipy.stats.multivariate_normal.logpdf(samples, mean=mean, cov=covariance)
            class_probabilities[cls] = class_probability

        return class_probabilities

    def _predict(self, samples, params=None):
        class_probabilities = self._calculate_class_probabilities(samples, self.class_means, self.class_covariances)

        # Get the class with the highest probability
        prob_array = jnp.array(list(class_probabilities.values()))
        # Filter out NaN values
        prob_array = jnp.nan_to_num(prob_array, nan=-jnp.inf)

        predictions = jnp.argmax(prob_array, axis=0)

        return  predictions, jnp.array([x.tolist() for x in class_probabilities.values()])

    def _set_label_offset(self, label_offset):
        self.label_offset = label_offset

    def pre_processing(self, features, labels, n_components=100):
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            features = self.pca.fit_transform(features)
        else:
            features = self.pca.transform(features)
        return features, labels



    def test(self, test_features, test_labels, current_epoch):

        test_features, test_labels = self.pre_processing(test_features, test_labels)

        # create a DataLoader for the test features and labels
        correct = 0

        # Predict the classes of the training set
        predictions, class_probabilities = self._predict(test_features)

        correct = jnp.sum(predictions == test_labels).tolist()


        print('Test set: Accuracy: {}/{} ({}%)'.format(
            correct, len(test_features), 100. * correct / float(len(test_features))
        ))
        # log the test accuracy to tensorboard
        self.writer.add_scalar("Accuracy/test", 100. * correct / float(len(test_features)), current_epoch) 


    def run(self, features, labels, test_features, test_labels):
        self.train(features, labels,1)
        self.test(test_features, test_labels, 1)

        # save the class means , covariances and supports to numpy files
        save(f"{self.save_dir}class_means.npy", np.array(self.class_means))
        save(f"{self.save_dir}class_covariances.npy", np.array(self.class_covariances))
        save(f"{self.save_dir}class_supports.npy", np.array(self.class_supports))

        # save pca model of sklearn, so that it can be used to transform the features in the future
        joblib.dump(self.pca, f"{self.save_dir}pca_model.joblib")
