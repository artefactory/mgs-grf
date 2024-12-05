import math

import numpy as np
from imblearn.over_sampling import SMOTE

from imblearn.over_sampling.base import BaseOverSampler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import ledoit_wolf,oas,empirical_covariance
from imblearn.utils import check_target_type
from collections import Counter

class CVSmoteModel(object):
    """
    CVSmoteModel. It's an estimator and not a oversampling strategy only like the others class in this file.
    """

    def __init__(self, splitter, model, list_k_max=100, list_k_step=10):
        """_summary_

        Parameters
        ----------
        splitter : sk-learn spliter object (or child)
            _description_
        model : _type_
            _description_
        list_k_max : int, optional
            _description_, by default 100
        list_k_step : int, optional
            _description_, by default 10
        """
        self.splitter = splitter
        self.list_k_max = list_k_max  # why is it called list ?
        self.list_k_step = list_k_step  # why is it called list ?
        self.model = model
        self.estimators_ = [0]  # are you sure about it ?

    def fit(self, X, y, sample_weight=None):
        """
        X and y are numpy arrays
        sample_weight is a numpy array
        """

        n_positifs = np.array(y, dtype=bool).sum()
        list_k_neighbors = [
            5,
            max(int(0.01 * n_positifs), 1),
            max(int(0.1 * n_positifs), 1),
            max(int(np.sqrt(n_positifs)), 1),
            max(int(0.5 * n_positifs), 1),
            max(int(0.7 * n_positifs), 1),
        ]
        list_k_neighbors.extend(
            list(np.arange(1, self.list_k_max, self.list_k_step, dtype=int))
        )

        best_score = -1
        folds = list(
            self.splitter.split(X, y)
        )  # you really need to transform it into a list ?
        for k in list_k_neighbors:
            scores = []
            for train, test in folds:
                new_X, new_y = SMOTE(k_neighbors=k).fit_resample(X[train], y[train])
                self.model.fit(X=new_X, y=new_y, sample_weight=sample_weight)
                scores.append(
                    roc_auc_score(y[test], self.model.predict_proba(X[test])[:, 1])
                )
            if sum(scores) > best_score:
                best_k = k

        new_X, new_y = SMOTE(k_neighbors=best_k).fit_resample(X, y)
        self.model.fit(X=new_X, y=new_y, sample_weight=sample_weight)
        if hasattr(self.model, "estimators_"):
            self.estimators_ = self.model.estimators_

    def predict(self, X):
        """
        predict
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        predict_probas
        """
        return self.model.predict_proba(X)


class MGS(BaseOverSampler):
    """
    MGS : Multivariate Gaussian SMOTE
    """

    def __init__(
        self, K, n_points=None, llambda=1.0, sampling_strategy="auto", random_state=None
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        if n_points is None:
            self.n_points = K
        else:
            self.n_points = n_points
        self.random_state = random_state

    def _fit_resample(self, X, y=None, n_final_sample=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        if n_final_sample=None, objective is balanced data.
        """

        if y is None:
            X_positifs = X
            X_negatifs = np.ones((0, X.shape[1]))
            assert (
                n_final_sample is not None
            ), "You need to provide a number of final samples."
        else:
            X_positifs = X[y == 1]
            X_negatifs = X[y == 0]
            if n_final_sample is None:
                n_final_sample = (y == 0).sum()

        n_minoritaire = X_positifs.shape[0]
        dimension = X.shape[1]
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbor_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=False
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, dimension))
        np.random.seed(self.random_state)
        for i in range(n_synthetic_sample):
            indice = np.random.randint(n_minoritaire)
            indices_neigh = [
                0
            ]  ## the central point is selected for the expectation and covariance matrix
            indices_neigh.extend(
                #random.sample(range(1, self.K + 1), self.n_points)
                np.random.choice(a=range(1, self.K + 1),size=self.n_points)
            )  # The nearrest neighbor selected for the estimation
            indice_neighbors = neighbor_by_index[indice][indices_neigh]
            mu = (1 / self.K + 1) * X_positifs[indice_neighbors, :].sum(axis=0)
            sigma = (
                self.llambda
                * (1 / self.K + 1)
                * (X_positifs[indice_neighbors, :] - mu).T.dot(
                    (X_positifs[indice_neighbors, :] - mu)
                )
            )

            new_observation = np.random.multivariate_normal(
                mu, sigma, check_valid="raise"
            ).T
            new_samples[i, :] = new_observation
        np.random.seed()

        oversampled_X = np.concatenate((X_negatifs, X_positifs, new_samples), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y


class MGS2(BaseOverSampler):
    """
    MGS2 : Faster version of MGS using SVD decomposition
    """

    def __init__(
        self,
        K,
        llambda=1.0,
        sampling_strategy="auto",
        random_state=None,
        weighted_cov=False,
        kind_sampling='cholesky'
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        self.random_state = random_state
        self.weighted_cov = weighted_cov
        self.kind_sampling=kind_sampling

    def _fit_resample(self, X, y=None, n_final_sample=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        if n_final_sample=None, objective is balanced data.
        """

        if y is None:
            X_positifs = X
            X_negatifs = np.ones((0, X.shape[1]))
            assert (
                n_final_sample is not None
            ), "You need to provide a number of final samples."
        else:
            X_positifs = X[y == 1]
            X_negatifs = X[y == 0]
            if n_final_sample is None:
                n_final_sample = (y == 0).sum()

        n_minoritaire = X_positifs.shape[0]
        dimension = X.shape[1]
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbors_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=False
        )

        n_synthetic_sample = n_final_sample - n_minoritaire

        # computing mu and covariance at once for every minority class points
        all_neighbors = X_positifs[neighbors_by_index.flatten()]
        if self.weighted_cov:
            # We sample from central point
            mus = X_positifs
        else:
            # We sample from mean of neighbors
            mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                len(X_positifs), self.K + 1, dimension
            ).sum(axis=1)
        centered_X = X_positifs[neighbors_by_index.flatten()] - np.repeat(
            mus, self.K + 1, axis=0
        )
        centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension)

        if self.weighted_cov:
            distances = (centered_X**2).sum(axis=-1)
            distances[distances > 1e-10] = distances[distances > 1e-10] ** -0.25

            # inv sqrt for positives only and half of power for multiplication below
            distances /= distances.sum(axis=-1)[:, np.newaxis]
            centered_X = (
                np.repeat(distances[:, :, np.newaxis] ** 0.5, dimension, axis=2)
                * centered_X
            )

        covs = (
            self.llambda
            * np.matmul(np.swapaxes(centered_X, 1, 2), centered_X)
            / (self.K + 1)
        )

        if self.kind_sampling == 'svd':
        # spectral decomposition of all covariances
            eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
            eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
            As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]
        elif self.kind_sampling == 'cholesky' :
            As = np.linalg.cholesky(
                covs + 1e-10 * np.identity(dimension)
            ) 
        else: 
            raise ValueError(
                    "kind_sampling of MGS not supported"
                    "Available values : 'cholescky','svd' "
                )
        np.random.seed(self.random_state)

        indices = np.random.randint(n_minoritaire, size=n_synthetic_sample)
        new_samples = np.zeros((n_synthetic_sample, dimension))
        for i, central_point in enumerate(indices):
            u = np.random.normal(loc=0, scale=1, size=dimension)
            new_observation = mus[central_point, :] + As[central_point].dot(u)
            new_samples[i, :] = new_observation
        np.random.seed()

        oversampled_X = np.concatenate((X_negatifs, X_positifs, new_samples), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y