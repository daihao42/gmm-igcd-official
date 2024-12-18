from math import log
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
import optax
from jax import random
import jax

from numpyro.infer.autoguide import AutoMultivariateNormal

from torch.utils.tensorboard import SummaryWriter

from numpy import save

from sklearn.decomposition import PCA

from collections import defaultdict

import joblib

import copy


#from jax import config
#config.update("jax_enable_x64", True)

from tqdm import tqdm

class MNGMMDiagClassifier():

    def __init__(self, num_samples, num_dim, num_classes, grid_bounds=(-10., 10.)):
        self.num_samples = num_samples
        self.num_dim = num_dim
        self.num_classes = num_classes
        print(f"num_classes: {num_classes}")

        self.pca = None

        self.global_params = None

        self.is_init = True

        self.label_offset = 0

        self.reset_optimizer = True

        self.train_init_lr = 0.001

        self.support_size = np.ones(num_classes) * 1e-3

    def update_dir_infos(self, log_dir = "logs/", save_dir = "saved_models/"):
        self.writer = SummaryWriter(log_dir)
        self.save_dir = save_dir

    def set_rng_key(self, rng_key):
        self.rng_key = rng_key

        
    def init_parameters(self, n_epochs = 10, lr= 0.1, log_dir = "logs/", save_dir = "saved_models/", batch_size = 256,
                        train_likelihood_sample=8, test_likelihood_sample=16, use_cuda = True, num_samples = None):
        if num_samples is not None:
            self.num_samples = num_samples

        self._use_cuda = use_cuda

        self.lr = lr

        self.batch_size = batch_size
        self.save_dir = save_dir

        self.writer = SummaryWriter(log_dir)

    def model(self, X, y=None, num_classes=2, global_params=None):
        num_features = X.shape[1]
        #num_classes = len(jnp.unique(y)) if y is not None else 2

        #print(f"num_classes: {num_classes}")
        #print(f"num_features: {num_features}")

        if global_params is None:
            class_means = numpyro.param("class_means", jnp.zeros((num_classes, num_features)))
            # Use a diagonal covariance to reduce complexity
            #class_covs = numpyro.param("class_covs", jnp.ones((num_classes, num_features)))
            #class_covs = numpyro.param("class_covs", jnp.stack([jnp.eye(num_features)] * num_classes))
            class_covs = numpyro.param("class_covs", jnp.ones((num_classes, num_features)))
        else :
            class_means = numpyro.param("class_means", global_params["class_means"])
            class_covs = numpyro.param("class_covs", global_params["class_covs"])
        
        with numpyro.plate("batch", X.shape[0], subsample_size=128) as ind:
            X_batch = X[ind]
            y_batch = y[ind] if y is not None else None
            
            if y_batch is not None:
                base_dist = dist.MultivariateNormal(class_means[y_batch], jax.vmap(jnp.diag)(class_covs[y_batch]))
                numpyro.sample("obs", base_dist, obs=X_batch)


    def run_inference(self, X, y, test_X, test_y, init_lr=None, num_steps=1000):

        if init_lr is None:
            init_lr = 1e-4

        scheduler = optax.join_schedules(
            schedules=[
                optax.linear_schedule(init_value=init_lr, end_value=init_lr*10, transition_steps=100),
                optax.exponential_decay(init_value=init_lr*10, transition_steps=500, decay_rate=0.85),
            ],
            boundaries=[100]
        )

        self.guide = lambda *args, **kwargs: None

        if(self.is_init):

            print("Initializing model")

            optimizer = numpyro.optim.optax_to_numpyro(optax.adam(scheduler))

            self.svi = SVI(self.model, guide=self.guide, optim=optimizer, loss=Trace_ELBO())

            self.svi_state = self.svi.init(random.PRNGKey(0), X = X, y= y, num_classes=self.num_classes, global_params=self.global_params)

            self.is_init = False

        else:

            if self.reset_optimizer:

                print("Resetting optimizer")

                optimizer = numpyro.optim.optax_to_numpyro(optax.adam(scheduler))

                self.svi.optim = optimizer
        
        for step in tqdm(range(num_steps)):
            self.svi_state, loss = self.svi.update(self.svi_state, X= X, y=y, num_classes=self.num_classes)

            self.writer.add_scalar("Accuracy/train", loss.item(), step) 

            if step % 100 == 0:
                correct, total, acc = self.calculate_acc(self.svi.get_params(self.svi_state), X, y)
                correct_test, total_test, acc_test = self.calculate_acc(self.svi.get_params(self.svi_state), test_X, test_y)

                print(f"Step {step}: loss = {loss:.4f}, train_acc = {correct}/{total}, {acc:.2f}%, test_acc = {correct_test}/{total_test}, {acc_test:.2f}%")
                
            # Early stopping (pseudo-code)
            if step > 100 and abs(prev_loss - loss) < 1e-3:
                print("Early stopping due to convergence.")
                break
            prev_loss = loss

        return self.svi.get_params(self.svi_state)


    def pre_processing(self, features, labels, n_components=100):
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            features = self.pca.fit_transform(features)
        else:
            features = self.pca.transform(features)
        return features, labels


    def train(self, features, labels, test_features, test_labels, current_epoch, num_steps):
        features, labels = self.pre_processing(features, labels)
        test_features, test_labels = self.pre_processing(test_features, test_labels)

        labels = labels.astype(int)

        self.params = self.run_inference(jnp.array(features), jnp.array(labels), 
                                         jnp.array(test_features), jnp.array(test_labels), num_steps=num_steps , init_lr=5e-4)

        pred_labels, _ =self._predict(jnp.array(features), self.params)

        if self.global_params is not None:

            self.is_init = True

            # count the occurrences of each label
            label_counts = jnp.bincount(pred_labels)

            print(f"label_counts: {label_counts}")

            novel_idx = pred_labels >= self.label_offset

            features = features[novel_idx]
            labels = labels[novel_idx]
            print(f"novel_features: {features.shape}")

            # filter the labels with counts less than 100
            #filtered_labels = jnp.where(label_counts >= features.shape[1])[0]

            # print the filtered labels
            #filtered_labels = pred_labels[jnp.isin(pred_labels, filtered_labels)]
            #print(filtered_labels.shape)
            #filtered_features = features[jnp.isin(pred_labels, filtered_labels)]

            #self.params = self.run_inference(jnp.array(filtered_features), jnp.array(filtered_labels), init_lr=6e-4)
            # for happy backbone
            #self.params = self.run_inference(jnp.array(filtered_features), jnp.array(filtered_labels), init_lr=4e-5)
            # for happy backbone s3
            # also for new dino
            #self.params = self.run_inference(jnp.array(filtered_features), jnp.array(filtered_labels), init_lr=4e-5)
            self.params = self.run_inference(jnp.array(features), jnp.array(labels), \
                                             jnp.array(test_features), jnp.array(test_labels), init_lr=3.8e-4, num_steps=1000)
        
        correct, total, acc = self.calculate_acc(self.params, features, labels)

        print('Train set: Accuracy: {}/{} ({}%)'.format(correct, total, acc))

        self.global_params = copy.deepcopy(self.params)


    def calculate_acc(self, params, test_features, test_labels):

        pred_test_labels, _ = self._predict(jnp.array(test_features), params)

        correct = jnp.sum(pred_test_labels == test_labels).tolist()

        
        return correct, len(test_features), 100. * (correct / float(len(test_features)))

    def test(self, test_features, test_labels, current_epoch):
        test_features, test_labels = self.pre_processing(test_features, test_labels)

        pred_test_labels, _ = self._predict(jnp.array(test_features), self.params)

        correct = jnp.sum(pred_test_labels == test_labels).tolist()

        print('Test set: Accuracy: {}/{} ({}%)'.format(
            correct, len(test_features), 100. * correct / float(len(test_features))
        ))


    # output the acc of training data
    def _predict(self, X, params):
        class_means = params["class_means"]
        class_covs = params["class_covs"]
        log_probs = []

        for i in range(class_means.shape[0]):
            mvn = dist.MultivariateNormal(class_means[i], jnp.diag(class_covs[i]))
            log_probs.append(mvn.log_prob(X))
        
        log_probs = jnp.stack(log_probs, axis=-1)
        return jnp.argmax(log_probs, axis=-1), log_probs


    def _set_label_offset(self, label_offset):
        self.label_offset = label_offset


    def run(self, features, labels, test_features, test_labels, num_steps=1000):
        self.train(features, labels, test_features, test_labels, 1, num_steps)

        test_features, test_labels = self.pre_processing(test_features, test_labels)
        correct, total, acc = self.calculate_acc(self.params, test_features, test_labels)

        print('Test set: Accuracy: {}/{} ({}%)'.format(
            correct, total, acc
        ))

        # save the class means , covariances and supports to numpy files
        save(f"{self.save_dir}class_means.npy", np.array(self.params["class_means"]))
        save(f"{self.save_dir}class_covariances.npy", np.array(self.params["class_covs"]))

        
