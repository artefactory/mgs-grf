import random
import math
from typing import Optional, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE

from imblearn.over_sampling.base import BaseOverSampler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from scipy import stats
from sklearn.preprocessing import OneHotEncoder
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
                random.sample(range(1, self.K + 1), self.n_points)
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
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        self.random_state = random_state
        self.weighted_cov = weighted_cov

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

        # spectral decomposition of all covariances
        # eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
        # eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
        # As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]
        As = np.linalg.cholesky(
            covs + 1e-10 * np.identity(dimension)
        )  ## add parameter for 1e-10 ?

        np.random.seed(self.random_state)
        # sampling all new points
        # u = np.random.normal(loc=0, scale=1, size=(len(indices), dimension))
        # new_samples = [mus[central_point] + As[central_point].dot(u[central_point]) for i in indices]
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


class MGSY(BaseOverSampler):
    """Class to perform MGS.

    Oversample the minority class by generating synthetic samples
    generated from a Gaussian distribution centered around the mean of nearest neighbors.

    Parameters
    ----------
    K : int
        The number of nearest neighbors to consider.
    llambda : float
        Controls the dilation of the covariance matrix.
    sampling_strategy : str, default='auto'
        The sampling strategy to use.
    random_state : Optional[int], default=None
        Controls the randomization of the algorithm.
    """

    def __init__(
        self,
        K: int,
        llambda: float,
        kind_sampling="cholesky",
        batch_size_sampling: Optional[int] = None,
        sampling_strategy="auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        self.kind_sampling = kind_sampling
        self.batch_size_sampling = batch_size_sampling
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def _fit_resample(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample the dataset.

        Parameters
        ----------
        X : np.ndarray
            The input data.
        y : Optional[np.ndarray]
            The target labels.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            The resampled data and target.
        """
        # Binary classification we only resample once (and we assume 0 - 1 classification)
        class_sample, num_samples = self.sampling_strategy_.popitem(last=False)

        X_minority = X[y == class_sample]
        X_majority = X[y != class_sample]
        n_minority = X_minority.shape[0]
        dimension = X.shape[1]

        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_minority)

        indices = self._rng.choice(n_minority, size=num_samples)
        neighbors_by_index = neigh.kneighbors(
            X=X_minority[indices], n_neighbors=self.K + 1, return_distance=False
        )

        mus, As = self._compute_mu_and_cov(
            X_minority, neighbors_by_index, dimension, num_samples
        )
        new_samples = self._generate_samples(
            num_samples, n_minority, dimension, mus, As
        )

        # concat
        oversampled_X = np.concatenate((X_majority, X_minority, new_samples), axis=0)
        oversampled_y = np.hstack(
            (
                np.full(len(X_majority), 1 - class_sample),
                np.full(n_minority + num_samples, class_sample),
            )
        )
        return oversampled_X, oversampled_y

    def _compute_mu_and_cov(
        self,
        X_minority: np.ndarray,
        neighbors_by_index: np.ndarray,
        dimension: int,
        num_samples,
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_neighbors = X_minority[neighbors_by_index.flatten()]
        all_neighbors_reshaped = all_neighbors.reshape(
            num_samples, self.K + 1, dimension
        )
        mus = np.mean(all_neighbors_reshaped, axis=1)
        centered_X = X_minority[neighbors_by_index.flatten()] - np.repeat(
            mus, self.K + 1, axis=0
        )
        centered_X = centered_X.reshape(num_samples, self.K + 1, dimension)
        covs = (
            self.llambda
            * np.matmul(np.swapaxes(centered_X, 1, 2), centered_X)
            / (self.K + 1)
        )

        if self.kind_sampling == "cholesky":
            As = np.linalg.cholesky(
                covs + (1e-10) * np.identity(dimension)
            )  ## add parameter for 1e-10 ?
        elif self.kind_sampling == "svd":
            eigen_values, eigen_vectors = np.linalg.eigh(covs)
            eigen_values[eigen_values > 1e-10] = (
                eigen_values[eigen_values > 1e-10] ** 0.5
            )
            As = [
                eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))
            ]
        else:
            raise ValueError(
                "kind_sampling of MGS not supported" "Available values : cholescky,svd "
            )
        return mus, As

    def _generate_samples(
        self,
        n_synthetic_sample: int,
        n_minority: int,
        dimension: int,
        mus: np.ndarray,
        As: np.ndarray,
    ) -> np.ndarray:
        if self.batch_size_sampling is None:  ## Case no loop
            u = self._rng.normal(loc=0, scale=1, size=(n_synthetic_sample, dimension))
            # indices = np.random.randint(n_minority,size=n_synthetic_sample)
            new_samples = [
                mus[central_point] + As[central_point].dot(u[central_point])
                for central_point in range(n_synthetic_sample)
            ]
            new_samples = np.array(new_samples)

        elif self.batch_size_sampling == 1:  # Case with loop
            new_samples = np.zeros((n_synthetic_sample, dimension))
            for i in range(n_synthetic_sample):
                u = self._rng.normal(loc=0, scale=1, size=dimension)
                new_observation = mus[i, :] + As[i].dot(u)
                new_samples[i, :] = new_observation

        else:  # Case batch loop
            n_batch = n_synthetic_sample // self.batch_size_sampling
            new_samples = np.zeros((n_synthetic_sample, dimension))
            for i in range(n_batch):
                u = self._rng.normal(
                    loc=0, scale=1, size=(self.batch_size_sampling, dimension)
                )
                actual_batch_indices = np.arange(
                    start=i * self.batch_size_sampling,
                    stop=(i + 1) * self.batch_size_sampling,
                    step=1,
                )
                new_observations = [
                    mus[central_point] + As[central_point].dot(u[ind])
                    for ind, central_point in enumerate(actual_batch_indices)
                ]
                new_samples[actual_batch_indices, :] = new_observations
            # rest :
            n_rest = n_synthetic_sample % self.batch_size_sampling
            u = self._rng.normal(loc=0, scale=1, size=(n_rest, dimension))
            actual_batch_indices = np.arange(
                start=n_batch * self.batch_size_sampling,
                stop=n_synthetic_sample,
                step=1,
            )
            new_observations = [
                mus[central_point] + As[central_point].dot(u[ind])
                for ind, central_point in enumerate(actual_batch_indices)
            ]
            new_samples[-n_rest:, :] = new_observations
        return new_samples


class MGSWeighted(MGSY):
    """Class to perform MGS with weighted covariance matrix.

    Oversample the minority class by generating synthetic samples
    generated from a Gaussian distribution. Covariance matrix is weighted by nearest neighbors distances to the mean point.

    Parameters
    ----------
    K : int
        The number of nearest neighbors to consider.
    llambda : float
        Controls the dilation of the covariance matrix.
    sampling_strategy : str, default='auto'
        The sampling strategy to use.
    random_state : Optional[int], default=None
        Controls the randomization of the algorithm.

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        X_resampled : array-like of shape (n_samples_new, n_features)
            The resampled data.
        y_resampled : array-like of shape (n_samples_new,)
            The resampled target.
    """

    def __init__(
        self,
        K: int,
        llambda: float,
        kind_sampling="cholesky",
        batch_size_sampling: Optional[int] = None,
        sampling_strategy: str = "auto",
        random_state: Optional[int] = None,
    ):
        super().__init__(
            K,
            llambda,
            kind_sampling,
            batch_size_sampling,
            sampling_strategy,
            random_state,
        )
        self._rng = np.random.default_rng(random_state)

    def _compute_mu_and_cov(
        self, X_positives: np.ndarray, neighbors_by_index: np.ndarray, dimension: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_neighbors = X_positives[neighbors_by_index.flatten()]
        all_neighbors_reshaped = all_neighbors.reshape(
            len(X_positives), self.K + 1, dimension
        )
        mus = np.mean(all_neighbors_reshaped, axis=1)
        centered_X = X_positives[neighbors_by_index.flatten()] - np.repeat(
            mus, self.K + 1, axis=0
        )
        centered_X = centered_X.reshape(len(X_positives), self.K + 1, dimension)

        epsilon = 1e-6
        diff = all_neighbors_reshaped - X_positives.reshape(
            len(X_positives), 1, X_positives.shape[1]
        )
        distances = np.linalg.norm(diff, axis=2)
        inverse_distances = 1 / (distances + epsilon)
        weights = inverse_distances / (np.sum(inverse_distances, axis=1).reshape(-1, 1))
        n = X_positives.shape[0]
        m = self.K + 1

        diag_matrices = np.zeros((n, m, m))
        diag_matrices[np.arange(n)[:, None], np.arange(m), np.arange(m)] = weights
        covs = self.llambda * np.swapaxes(centered_X, 1, 2) @ diag_matrices @ centered_X

        if self.kind_sampling == "cholescky":
            As = np.linalg.cholesky(
                covs + (1e-10) * np.identity(dimension)
            )  ## add parameter for 1e-10 ?
        elif self.kind_sampling == "svd":
            eigen_values, eigen_vectors = np.linalg.eigh(covs)
            eigen_values[eigen_values > 1e-10] = (
                eigen_values[eigen_values > 1e-10] ** 0.5
            )
            As = [
                eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))
            ]
        else:
            raise ValueError(
                "kind_sampling of MGS not supported" "Available values : cholescky,svd "
            )

        return mus, As


class NoSampling(object):
    """
    None rebalancing strategy class
    """

    def fit_resample(self, X, y):
        """
        X is a numpy array
        y is a numpy array of dimension (n,)
        """
        return X, y


##########################################
######## CATEGORICAL #####################
#########################################
class WMGS_NC(BaseOverSampler):
    """
    MGS NC strategy
    """

    def __init__(
        self,
        K,
        categorical_features,
        version,
        weighted_cov=True,
        n_points=None,
        llambda=1.0,
        sampling_strategy="auto",
        random_state=None,
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
        self.categorical_features = categorical_features
        self.version = version
        self.weighted_cov = weighted_cov
        self.random_state = random_state

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        # X = _check_X(X)
        # self._check_n_features(X, reset=True)
        # self._check_feature_names(X, reset=True)
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.categorical_features_.size == 0:
            raise ValueError(
                "MGS-NC is not designed to work only with numerical "
                "features. It requires some categorical features."
            )

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

        if len(self.categorical_features) == X.shape[1]:
            raise ValueError(
                "MGS-NC is not designed to work only with categorical "
                "features. It requires some numerical features."
            )

        bool_mask = np.ones((X_positifs.shape[1]), dtype=bool)
        bool_mask[self.categorical_features] = False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:, bool_mask]  ## continuous features
        X_negatifs = X_negatifs_all_features[:, bool_mask]  ## continuous features
        X_positifs_categorical = X_positifs_all_features[:, ~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:, ~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension_continuous = X_positifs.shape[1]  ## of continuous features

        enc = OneHotEncoder(handle_unknown="ignore")  ## encoding
        X_positifs_categorical_enc = enc.fit_transform(
            X_positifs_categorical
        ).toarray()
        X_positifs_all_features_enc = np.hstack((X_positifs,X_positifs_categorical_enc))
        cste_med = np.median(
            np.sqrt(np.var(X_positifs, axis=0))
        )  ## med constante from continuous variables
        if not math.isclose(cste_med, 0):
            X_positifs_all_features_enc[:, dimension_continuous:] = cste_med / np.sqrt(
                2
            )  # With one-hot encoding, the median will be repeated twice. We need
        # to divide by sqrt(2) such that we only have one median value
        # contributing to the Euclidean distance
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs_all_features_enc)
        neighbor_by_dist, neighbor_by_index = neigh.kneighbors(
            X=X_positifs_all_features_enc, n_neighbors=self.K + 1, return_distance=True
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        # computing mu and covariance at once for every minority class points
        all_neighbors = X_positifs[neighbor_by_index.flatten()]
        if self.weighted_cov:
            # We sample from central point
            mus = X_positifs
        else:
            # We sample from mean of neighbors
            mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                len(X_positifs), self.K + 1, dimension_continuous
            ).sum(axis=1)
        centered_X = X_positifs[neighbor_by_index.flatten()] - np.repeat(
            mus, self.K + 1, axis=0
        )
        centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension_continuous)

        if self.weighted_cov:
            distances = (centered_X**2).sum(axis=-1)
            distances[distances > 1e-10] = distances[distances > 1e-10] ** -0.25

            # inv sqrt for positives only and half of power for multiplication below
            distances /= distances.sum(axis=-1)[:, np.newaxis]
            centered_X = (
                np.repeat(distances[:, :, np.newaxis] ** 0.5, dimension_continuous, axis=2)
                * centered_X
            )

        covs = (
            self.llambda
            * np.matmul(np.swapaxes(centered_X, 1, 2), centered_X)
            / (self.K + 1)
        )

        # spectral decomposition of all covariances
        # eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
        # eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
        # As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]
        As = np.linalg.cholesky(
            covs + 1e-10 * np.identity(dimension_continuous)
        )  ## add parameter for 1e-10 ?

        np.random.seed(self.random_state)
        # sampling all new points
        # u = np.random.normal(loc=0, scale=1, size=(len(indices), dimension))
        # new_samples = [mus[central_point] + As[central_point].dot(u[central_point]) for i in indices]
        indices = np.random.randint(n_minoritaire, size=n_synthetic_sample)
        new_samples = np.zeros((n_synthetic_sample, dimension_continuous))
        for i, central_point in enumerate(indices):
            u = np.random.normal(loc=0, scale=1, size=dimension_continuous)
            new_observation = mus[central_point, :] + As[central_point].dot(u)
            new_samples[i, :] = new_observation
            ############### CATEGORICAL ##################
        new_samples_cat = np.zeros(
            (n_synthetic_sample, len(self.categorical_features)), dtype=object
        )
        for i in range(n_synthetic_sample):
            indice = np.random.randint(n_minoritaire)
            indices_neigh = [
                0
            ]  ## the central point is selected for the expectation and covariance matrix
            indices_neigh.extend(
                random.sample(range(1, self.K + 1), self.n_points)
            )  # The nearrest neighbor selected for the estimation
            indice_neighbors = neighbor_by_index[indice][indices_neigh]

            if (
                self.version == 1
            ):  ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    most_common = Counter(
                        X_positifs_categorical[indice_neighbors, cat_feature]
                    ).most_common(1)[0][0]
                    new_samples_cat[i, cat_feature] = most_common
            elif (
                self.version == 2
            ):  ## sampling of one of the nearest neighbors per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors, cat_feature],
                        replace=False,
                    )
            elif (
                self.version == 3
            ):  ## sampling of one of the nearest neighbors per categorical feature using dsitance
                #### We take the nn of the central point. The latter is excluded
                epsilon_weigths_sampling = 10e-6
                indice_neighbors_without_0 = np.arange(
                    start=1, stop=self.K + 1, dtype=int
                )
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors_without_0, cat_feature],
                        replace=False,
                        p=(
                            (
                                1
                                / (
                                    neighbor_by_dist[indice][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            )
                            / (
                                1
                                / (
                                    neighbor_by_dist[indice][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            ).sum()
                        ),
                    )
            else:
                raise ValueError(
                    "Selected version not allowed " "Please chose an existing version"
                )
        np.random.seed()

        ##### END ######
        new_samples_final = np.zeros(
            (n_synthetic_sample, X_positifs_all_features.shape[1]), dtype=object
        )
        new_samples_final[:, bool_mask] = new_samples
        new_samples_final[:, ~bool_mask] = new_samples_cat

        X_positifs_final = np.zeros(
            (len(X_positifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_positifs_final[:, bool_mask] = X_positifs
        X_positifs_final[:, ~bool_mask] = X_positifs_categorical

        X_negatifs_final = np.zeros(
            (len(X_negatifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_negatifs_final[:, bool_mask] = X_negatifs
        X_negatifs_final[:, ~bool_mask] = X_negatifs_categorical

        oversampled_X = np.concatenate(
            (X_negatifs_final, X_positifs_final, new_samples_final), axis=0
        )
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y


class WMGS_NC_cov(BaseOverSampler):
    """
    MGS NC strategy
    """

    def __init__(
        self,
        K,
        categorical_features,
        version,
        weighted_cov=True,
        ledoitwolfcov = False,
        oascov=False,
        tracecov=False,
        idcov=False,
        expcov=False,
        mucentered=False,
        n_points=None,
        llambda=1.0,
        sampling_strategy="auto",
        random_state=None,
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
        self.categorical_features = categorical_features
        self.version = version
        self.weighted_cov = weighted_cov
        self.ledoitwolfcov=ledoitwolfcov
        self.oascov=oascov
        self.tracecov=tracecov
        self.idcov=idcov
        self.expcov=expcov
        self.mucentered=mucentered
        self.random_state = random_state

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        # X = _check_X(X)
        # self._check_n_features(X, reset=True)
        # self._check_feature_names(X, reset=True)
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.categorical_features_.size == 0:
            raise ValueError(
                "MGS-NC is not designed to work only with numerical "
                "features. It requires some categorical features."
            )

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

        if len(self.categorical_features) == X.shape[1]:
            raise ValueError(
                "MGS-NC is not designed to work only with categorical "
                "features. It requires some numerical features."
            )

        bool_mask = np.ones((X_positifs.shape[1]), dtype=bool)
        bool_mask[self.categorical_features] = False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:, bool_mask]  ## continuous features
        X_negatifs = X_negatifs_all_features[:, bool_mask]  ## continuous features
        X_positifs_categorical = X_positifs_all_features[:, ~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:, ~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension_continuous = X_positifs.shape[1]  ## of continuous features

        enc = OneHotEncoder(handle_unknown="ignore")  ## encoding
        X_positifs_categorical_enc = enc.fit_transform(
            X_positifs_categorical
        ).toarray()
        X_positifs_all_features_enc = np.hstack((X_positifs,X_positifs_categorical_enc))
        cste_med = np.median(
            np.sqrt(np.var(X_positifs, axis=0))
        )  ## med constante from continuous variables
        if not math.isclose(cste_med, 0):
            X_positifs_all_features_enc[:, dimension_continuous:] = cste_med / np.sqrt(
                2
            )  # With one-hot encoding, the median will be repeated twice. We need
        # to divide by sqrt(2) such that we only have one median value
        # contributing to the Euclidean distance
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs_all_features_enc)
        neighbor_by_dist, neighbor_by_index = neigh.kneighbors(
            X=X_positifs_all_features_enc, n_neighbors=self.K + 1, return_distance=True
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        if self.ledoitwolfcov:
            if self.mucentered:
                # We sample from mean of neighbors
                all_neighbors = X_positifs[neighbor_by_index.flatten()]
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension_continuous
                    ).sum(axis=1)
            else:
                 # We sample from central point
                mus = X_positifs
            As = []
            for i in range(n_minoritaire):
                covariance, shrinkage = ledoit_wolf(X_positifs[neighbor_by_index[i,1:],:]-mus[neighbor_by_index[i,0]],assume_centered=True)
                As.append(self.llambda*covariance)
            As= np.array(As)   
        
        elif self.oascov:
            if self.mucentered:
                # We sample from mean of neighbors
                all_neighbors = X_positifs[neighbor_by_index.flatten()]
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension_continuous
                    ).sum(axis=1)
            else:
                 # We sample from central point
                mus = X_positifs
            As = []
            for i in range(n_minoritaire):
                covariance, shrinkage = oas(X_positifs[neighbor_by_index[i,1:],:]-mus[neighbor_by_index[i,0]],assume_centered=True)
                As.append(self.llambda*covariance)
            As= np.array(As) 
        elif self.tracecov:
            if self.mucentered:
                # We sample from mean of neighbors
                all_neighbors = X_positifs[neighbor_by_index.flatten()]
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension_continuous
                    ).sum(axis=1)
            else:
                 # We sample from central point
                mus = X_positifs
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                covariance  = empirical_covariance(X_positifs[neighbor_by_index[i,1:],:]-mus[neighbor_by_index[i,0]],assume_centered=True)
                final_covariance = (np.trace(covariance)/p) * np.eye(p)
                As.append(self.llambda*final_covariance) 
            As= np.array(As) 
        elif self.idcov:
            if self.mucentered:
                # We sample from mean of neighbors
                all_neighbors = X_positifs[neighbor_by_index.flatten()]
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension_continuous
                    ).sum(axis=1)
            else:
                 # We sample from central point
                mus = X_positifs
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                final_covariance = (1/p) * np.eye(p)
                As.append(self.llambda*final_covariance) 
            As= np.array(As) 
        elif self.expcov:
            if self.mucentered:
                # We sample from mean of neighbors
                all_neighbors = X_positifs[neighbor_by_index.flatten()]
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension_continuous
                    ).sum(axis=1)
            else:
                 # We sample from central point
                mus = X_positifs
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                diffs = X_positifs[neighbor_by_index[i,1:],:]-mus[neighbor_by_index[i,0]]
                exp_dist = np.exp(-np.linalg.norm(diffs, axis=1))
                weights = exp_dist / (np.sum(exp_dist))
                final_covariance = (diffs.T.dot(np.diag(weights)).dot(diffs)) + np.eye(dimension_continuous) * 1e-10
                As.append(self.llambda*final_covariance) 
            As= np.array(As) 
            

        else:
            # computing mu and covariance at once for every minority class points
            all_neighbors = X_positifs[neighbor_by_index.flatten()]
            if self.weighted_cov:
                # We sample from central point
                mus = X_positifs
            else:
                # We sample from mean of neighbors
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension_continuous
                ).sum(axis=1)
            centered_X = X_positifs[neighbor_by_index.flatten()] - np.repeat(
                mus, self.K + 1, axis=0
            )
            centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension_continuous)

            if self.weighted_cov:
                distances = (centered_X**2).sum(axis=-1)
                distances[distances > 1e-10] = distances[distances > 1e-10] ** -0.25

                # inv sqrt for positives only and half of power for multiplication below
                distances /= distances.sum(axis=-1)[:, np.newaxis]
                centered_X = (
                    np.repeat(distances[:, :, np.newaxis] ** 0.5, dimension_continuous, axis=2)
                    * centered_X
                )

            covs = (
                self.llambda
                * np.matmul(np.swapaxes(centered_X, 1, 2), centered_X)
                / (self.K + 1)
            )

            # spectral decomposition of all covariances
            # eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
            # eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
            # As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]
            As = np.linalg.cholesky(
                covs + 1e-10 * np.identity(dimension_continuous)
            )  ## add parameter for 1e-10 ?

        np.random.seed(self.random_state)
        # sampling all new points
        # u = np.random.normal(loc=0, scale=1, size=(len(indices), dimension))
        # new_samples = [mus[central_point] + As[central_point].dot(u[central_point]) for i in indices]
        indices = np.random.randint(n_minoritaire, size=n_synthetic_sample)
        new_samples = np.zeros((n_synthetic_sample, dimension_continuous))
        for i, central_point in enumerate(indices):
            u = np.random.normal(loc=0, scale=1, size=dimension_continuous)
            new_observation = mus[central_point, :] + As[central_point].dot(u)
            new_samples[i, :] = new_observation
            ############### CATEGORICAL ##################
        new_samples_cat = np.zeros(
            (n_synthetic_sample, len(self.categorical_features)), dtype=object
        )
        for i in range(n_synthetic_sample):
            indice = np.random.randint(n_minoritaire)
            indices_neigh = [
                0
            ]  ## the central point is selected for the expectation and covariance matrix
            indices_neigh.extend(
                random.sample(range(1, self.K + 1), self.n_points)
            )  # The nearrest neighbor selected for the estimation
            indice_neighbors = neighbor_by_index[indice][indices_neigh]

            if (
                self.version == 1
            ):  ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    most_common = Counter(
                        X_positifs_categorical[indice_neighbors, cat_feature]
                    ).most_common(1)[0][0]
                    new_samples_cat[i, cat_feature] = most_common
            elif (
                self.version == 2
            ):  ## sampling of one of the nearest neighbors per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors, cat_feature],
                        replace=False,
                    )
            elif (
                self.version == 3
            ):  ## sampling of one of the nearest neighbors per categorical feature using dsitance
                #### We take the nn of the central point. The latter is excluded
                epsilon_weigths_sampling = 10e-6
                indice_neighbors_without_0 = np.arange(
                    start=1, stop=self.K + 1, dtype=int
                )
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors_without_0, cat_feature],
                        replace=False,
                        p=(
                            (
                                1
                                / (
                                    neighbor_by_dist[indice][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            )
                            / (
                                1
                                / (
                                    neighbor_by_dist[indice][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            ).sum()
                        ),
                    )
            else:
                raise ValueError(
                    "Selected version not allowed " "Please chose an existing version"
                )
        np.random.seed()

        ##### END ######
        new_samples_final = np.zeros(
            (n_synthetic_sample, X_positifs_all_features.shape[1]), dtype=object
        )
        new_samples_final[:, bool_mask] = new_samples
        new_samples_final[:, ~bool_mask] = new_samples_cat

        X_positifs_final = np.zeros(
            (len(X_positifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_positifs_final[:, bool_mask] = X_positifs
        X_positifs_final[:, ~bool_mask] = X_positifs_categorical

        X_negatifs_final = np.zeros(
            (len(X_negatifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_negatifs_final[:, bool_mask] = X_negatifs
        X_negatifs_final[:, ~bool_mask] = X_negatifs_categorical

        oversampled_X = np.concatenate(
            (X_negatifs_final, X_positifs_final, new_samples_final), axis=0
        )
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y
    

class WMGS_NC_cont(BaseOverSampler):
    """
    MGS NC strategy
    """

    def __init__(
        self,
        K,
        categorical_features,
        version,
        weighted_cov=True,
        n_points=None,
        llambda=1.0,
        sampling_strategy="auto",
        random_state=None,
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
        self.categorical_features = categorical_features
        self.version = version
        self.weighted_cov = weighted_cov
        self.random_state = random_state

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        # X = _check_X(X)
        # self._check_n_features(X, reset=True)
        # self._check_feature_names(X, reset=True)
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.categorical_features_.size == 0:
            raise ValueError(
                "MGS-NC is not designed to work only with numerical "
                "features. It requires some categorical features."
            )

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

        if len(self.categorical_features) == X.shape[1]:
            raise ValueError(
                "MGS-NC is not designed to work only with categorical "
                "features. It requires some numerical features."
            )

        bool_mask = np.ones((X_positifs.shape[1]), dtype=bool)
        bool_mask[self.categorical_features] = False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:, bool_mask]  ## continuous features
        X_negatifs = X_negatifs_all_features[:, bool_mask]  ## continuous features
        X_positifs_categorical = X_positifs_all_features[:, ~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:, ~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension_continuous = X_positifs.shape[1]  ## of continuous features

        enc = OneHotEncoder(handle_unknown="ignore")  ## encoding
        X_positifs_categorical_enc = enc.fit_transform(
            X_positifs_categorical
        ).toarray()
        X_positifs_all_features_enc = np.hstack((X_positifs,X_positifs_categorical_enc))
        cste_med = np.median(
            np.sqrt(np.var(X_positifs, axis=0))
        )  ## med constante from continuous variables
        if not math.isclose(cste_med, 0):
            X_positifs_all_features_enc[:, dimension_continuous:] = cste_med / np.sqrt(
                2
            )  # With one-hot encoding, the median will be repeated twice. We need
        # to divide by sqrt(2) such that we only have one median value
        # contributing to the Euclidean distance
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs_all_features_enc)
        neighbor_by_dist, neighbor_by_index = neigh.kneighbors(
            X=X_positifs_all_features_enc, n_neighbors=self.K + 1, return_distance=True
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        # computing mu and covariance at once for every minority class points
        all_neighbors = X_positifs[neighbor_by_index.flatten()]
        if self.weighted_cov:
            # We sample from central point
            mus = X_positifs
        else:
            # We sample from mean of neighbors
            mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                len(X_positifs), self.K + 1, dimension_continuous
            ).sum(axis=1)
        centered_X = X_positifs[neighbor_by_index.flatten()] - np.repeat(
            mus, self.K + 1, axis=0
        )
        centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension_continuous)

        if self.weighted_cov:
            distances = (centered_X**2).sum(axis=-1)
            distances[distances > 1e-10] = distances[distances > 1e-10] ** -0.25

            # inv sqrt for positives only and half of power for multiplication below
            distances /= distances.sum(axis=-1)[:, np.newaxis]
            centered_X = (
                np.repeat(distances[:, :, np.newaxis] ** 0.5, dimension_continuous, axis=2)
                * centered_X
            )

        covs = (
            self.llambda
            * np.matmul(np.swapaxes(centered_X, 1, 2), centered_X)
            / (self.K + 1)
        )

        # spectral decomposition of all covariances
        # eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
        # eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
        # As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]
        As = np.linalg.cholesky(
            covs + 1e-10 * np.identity(dimension_continuous)
        )  ## add parameter for 1e-10 ?

        np.random.seed(self.random_state)
        # sampling all new points
        # u = np.random.normal(loc=0, scale=1, size=(len(indices), dimension))
        # new_samples = [mus[central_point] + As[central_point].dot(u[central_point]) for i in indices]
        indices = np.random.randint(n_minoritaire, size=n_synthetic_sample)
        new_samples = np.zeros((n_synthetic_sample, dimension_continuous))
        for i, central_point in enumerate(indices):
            u = np.random.normal(loc=0, scale=1, size=dimension_continuous)
            new_observation = mus[central_point, :] + As[central_point].dot(u)
            new_samples[i, :] = new_observation
        np.random.seed()

        ##### END ######
        oversampled_X = np.concatenate((X_negatifs, X_positifs, new_samples), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )
        return oversampled_X, oversampled_y
    


class WMGS_encode(BaseOverSampler):
    """
    MGS NC strategy
    """

    def __init__(
        self,
        K,
        categorical_features,
        weighted_cov=True,
        n_points=None,
        llambda=1.0,
        sampling_strategy="auto",
        random_state=None,
        bool_encoding=None,
        categorical_features_one_hot=None
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
        self.categorical_features = categorical_features
        self.weighted_cov = weighted_cov
        self.random_state = random_state
        self.bool_encoding=bool_encoding # kind of encoding already applied to the data
        self.categorical_features_one_hot=categorical_features_one_hot

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        # X = _check_X(X)
        # self._check_n_features(X, reset=True)
        # self._check_feature_names(X, reset=True)
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.categorical_features_.size == 0:
            raise ValueError(
                "MGS-NC is not designed to work only with numerical "
                "features. It requires some categorical features."
            )

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

        if len(self.categorical_features) == X.shape[1]:
            raise ValueError(
                "MGS-NC is not designed to work only with categorical "
                "features. It requires some numerical features."
            )

        n_minoritaire = X_positifs.shape[0]
        dimension_continuous = X_positifs.shape[1]  ## of continuous features

        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbor_by_dist, neighbor_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=True
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        # computing mu and covariance at once for every minority class points
        all_neighbors = X_positifs[neighbor_by_index.flatten()]
        if self.weighted_cov:
            # We sample from central point
            mus = X_positifs
        else:
            # We sample from mean of neighbors
            mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                len(X_positifs), self.K + 1, dimension_continuous
            ).sum(axis=1)
        centered_X = X_positifs[neighbor_by_index.flatten()] - np.repeat(
            mus, self.K + 1, axis=0
        )
        centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension_continuous)

        if self.weighted_cov:
            distances = (centered_X**2).sum(axis=-1)
            distances[distances > 1e-10] = distances[distances > 1e-10] ** -0.25

            # inv sqrt for positives only and half of power for multiplication below
            distances /= distances.sum(axis=-1)[:, np.newaxis]
            centered_X = (
                np.repeat(distances[:, :, np.newaxis] ** 0.5, dimension_continuous, axis=2)
                * centered_X
            )

        covs = (
            self.llambda
            * np.matmul(np.swapaxes(centered_X, 1, 2), centered_X)
            / (self.K + 1)
        )

        # spectral decomposition of all covariances
        # eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
        # eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
        # As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]
        As = np.linalg.cholesky(
            covs + 1e-10 * np.identity(dimension_continuous)
        )  ## add parameter for 1e-10 ?

        np.random.seed(self.random_state)
        # sampling all new points
        # u = np.random.normal(loc=0, scale=1, size=(len(indices), dimension))
        # new_samples = [mus[central_point] + As[central_point].dot(u[central_point]) for i in indices]
        indices = np.random.randint(n_minoritaire, size=n_synthetic_sample)
        new_samples = np.zeros((n_synthetic_sample, dimension_continuous))
        for i, central_point in enumerate(indices):
            u = np.random.normal(loc=0, scale=1, size=dimension_continuous)
            new_observation = mus[central_point, :] + As[central_point].dot(u)
            if self.bool_encoding is None:
                new_samples[i, :] = new_observation
            elif self.bool_encoding=="one-hot":
                if self.categorical_features_one_hot is None : ## categorical_features_one_hot is a list of indexes of the modalities of categorical features
                    raise ValueError("No value set to categorical_features_one_hot")
                else:
                    for cols in self.categorical_features_one_hot:
                        argmaxes = np.argmax(new_observation[cols])
                        new_observation[cols] = np.zeros((len(cols),))
                        curr_cols = new_observation[cols]
                        curr_cols[argmaxes] = 1 ## argamx done on clos subset so we have to appky the argmax on new_observation[cols]
                        new_observation[cols] = curr_cols
                        new_samples[i, :] = new_observation 
            elif self.bool_encoding=="ordinal":
                new_observation[self.categorical_features] = np.rint(new_observation[self.categorical_features])
                new_samples[i, :] = new_observation 
            else:
                raise ValueError(
                    "Encoding not in list"
                    )
                
            
        np.random.seed()

        ##### END ######
        oversampled_X = np.concatenate((X_negatifs, X_positifs, new_samples), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )
        return oversampled_X, oversampled_y
    

class MGS_NC(BaseOverSampler):
    """
    MGS NC strategy
    """

    def __init__(
        self,
        K,
        categorical_features,
        version,
        n_points=None,
        llambda=1.0,
        sampling_strategy="auto",
        random_state=None,
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
        self.categorical_features = categorical_features
        self.version = version
        self.random_state = random_state

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        # X = _check_X(X)
        # self._check_n_features(X, reset=True)
        # self._check_feature_names(X, reset=True)
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.categorical_features_.size == 0:
            raise ValueError(
                "MGS-NC is not designed to work only with numerical "
                "features. It requires some categorical features."
            )

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

        if len(self.categorical_features) == X.shape[1]:
            raise ValueError(
                "MGS-NC is not designed to work only with categorical "
                "features. It requires some numerical features."
            )

        bool_mask = np.ones((X_positifs.shape[1]), dtype=bool)
        bool_mask[self.categorical_features] = False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:, bool_mask]  ## continuous features
        X_negatifs = X_negatifs_all_features[:, bool_mask]  ## continuous features
        X_positifs_categorical = X_positifs_all_features[:, ~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:, ~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension_continuous = X_positifs.shape[1]  ## of continuous features

        enc = OneHotEncoder(handle_unknown="ignore")  ## encoding
        X_positifs_all_features_enc = enc.fit_transform(
            X_positifs_all_features
        ).toarray()
        cste_med = np.median(
            np.sqrt(np.var(X_positifs, axis=0))
        )  ## med constante from continuous variables
        if not math.isclose(cste_med, 0):
            X_positifs_all_features_enc[:, dimension_continuous:] = cste_med / np.sqrt(
                2
            )  # With one-hot encoding, the median will be repeated twice. We need
        # to divide by sqrt(2) such that we only have one median value
        # contributing to the Euclidean distance
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs_all_features_enc)
        neighbor_by_dist, neighbor_by_index = neigh.kneighbors(
            X=X_positifs_all_features_enc, n_neighbors=self.K + 1, return_distance=True
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, dimension_continuous))
        new_samples_cat = np.zeros(
            (n_synthetic_sample, len(self.categorical_features)), dtype=object
        )

        np.random.seed(self.random_state)
        for i in range(n_synthetic_sample):
            ######### CONTINUOUS ################
            indice = np.random.randint(n_minoritaire)
            indices_neigh = [
                0
            ]  ## the central point is selected for the expectation and covariance matrix
            indices_neigh.extend(
                random.sample(range(1, self.K + 1), self.n_points)
            )  # The nearrest neighbor selected for the estimation
            indice_neighbors = neighbor_by_index[indice][indices_neigh]
            mu = (1 / self.K + 1) * X_positifs[indice_neighbors, :].sum(axis=0)
            sigma = (
                self.llambda
                * (1 / self.n_points)
                * (X_positifs[indice_neighbors, :] - mu).T.dot(
                    (X_positifs[indice_neighbors, :] - mu)
                )
            )
            new_observation = np.random.multivariate_normal(
                mu, sigma, check_valid="raise"
            ).T
            new_samples[i, :] = new_observation
            ############### CATEGORICAL ##################
            if (
                self.version == 1
            ):  ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    most_common = Counter(
                        X_positifs_categorical[indice_neighbors, cat_feature]
                    ).most_common(1)[0][0]
                    new_samples_cat[i, cat_feature] = most_common
            elif (
                self.version == 2
            ):  ## sampling of one of the nearest neighbors per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors, cat_feature],
                        replace=False,
                    )
            elif (
                self.version == 3
            ):  ## sampling of one of the nearest neighbors per categorical feature using dsitance
                #### We take the nn of the central point. The latter is excluded
                epsilon_weigths_sampling = 10e-6
                indice_neighbors_without_0 = np.arange(
                    start=1, stop=self.K + 1, dtype=int
                )
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors_without_0, cat_feature],
                        replace=False,
                        p=(
                            (
                                1
                                / (
                                    neighbor_by_dist[indice][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            )
                            / (
                                1
                                / (
                                    neighbor_by_dist[indice][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            ).sum()
                        ),
                    )
            else:
                raise ValueError(
                    "Selected version not allowed " "Please chose an existing version"
                )
        np.random.seed()

        ##### END ######
        new_samples_final = np.zeros(
            (n_synthetic_sample, X_positifs_all_features.shape[1]), dtype=object
        )
        new_samples_final[:, bool_mask] = new_samples
        new_samples_final[:, ~bool_mask] = new_samples_cat

        X_positifs_final = np.zeros(
            (len(X_positifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_positifs_final[:, bool_mask] = X_positifs
        X_positifs_final[:, ~bool_mask] = X_positifs_categorical

        X_negatifs_final = np.zeros(
            (len(X_negatifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_negatifs_final[:, bool_mask] = X_negatifs
        X_negatifs_final[:, ~bool_mask] = X_negatifs_categorical

        oversampled_X = np.concatenate(
            (X_negatifs_final, X_positifs_final, new_samples_final), axis=0
        )
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y


class MGS_cat(BaseOverSampler):
    """
    MGS NC distance-without-discrete-features
    """

    def __init__(
        self,
        K,
        categorical_features,
        version,
        n_points=None,
        llambda=10,
        sampling_strategy="auto",
        random_state=None,
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
        self.categorical_features = categorical_features
        self.version = version
        self.random_state = random_state

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.categorical_features_.size == 0:
            raise ValueError(
                "MGS_cat is not designed to work only with numerical "
                "features. It requires some categorical features."
            )

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
        if len(self.categorical_features) == X.shape[1]:
            raise ValueError(
                "MGS_cat is not designed to work only with categorical "
                "features. It requires some numerical features."
            )

        bool_mask = np.ones((X_positifs.shape[1]), dtype=bool)
        bool_mask[self.categorical_features] = False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:, bool_mask]  ## continuous features
        X_negatifs = X_negatifs_all_features[:, bool_mask]  ## continuous features
        X_positifs_categorical = X_positifs_all_features[:, ~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:, ~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension = X_positifs.shape[1]  ## features continues seulement

        ######### CONTINUOUS ################
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbor_by_dist, neighbor_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=True
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, dimension))
        new_samples_cat = np.zeros(
            (n_synthetic_sample, len(self.categorical_features)), dtype=object
        )
        np.random.seed(self.random_state)
        for i in range(n_synthetic_sample):
            indice = np.random.randint(n_minoritaire)
            indices_neigh = [
                0
            ]  ## the central point is selected for the expectation and covariance matrix
            indices_neigh.extend(
                random.sample(range(1, self.K + 1), self.n_points)
            )  # The nearrest neighbor selected for the estimation
            indice_neighbors = neighbor_by_index[indice][indices_neigh]
            mu = (1 / self.K + 1) * X_positifs[indice_neighbors, :].sum(axis=0)
            sigma = (
                self.llambda
                * (1 / self.n_points)
                * (X_positifs[indice_neighbors, :] - mu).T.dot(
                    (X_positifs[indice_neighbors, :] - mu)
                )
            )
            new_observation = np.random.multivariate_normal(
                mu, sigma, check_valid="raise"
            ).T
            new_samples[i, :] = new_observation
            ############### CATEGORICAL ##################
            if (
                self.version == 1
            ):  ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    most_common = Counter(
                        X_positifs_categorical[indice_neighbors, cat_feature]
                    ).most_common(1)[0][0]
                    new_samples_cat[i, cat_feature] = most_common
            elif (
                self.version == 2
            ):  ## sampling of one of the nearest neighbors per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors, cat_feature],
                        replace=False,
                    )
            elif (
                self.version == 3
            ):  ## sampling of one of the nearest neighbors per categorical feature using dsitance
                #### We take the nn of the central point. The latter is excluded
                epsilon_weigths_sampling = 10e-6
                indice_neighbors_without_0 = np.arange(
                    start=1, stop=self.K + 1, dtype=int
                )
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors_without_0, cat_feature],
                        replace=False,
                        p=(
                            (
                                1
                                / (
                                    neighbor_by_dist[indice][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            )
                            / (
                                1
                                / (
                                    neighbor_by_dist[indice][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            ).sum()
                        ),
                    )
            else:
                raise ValueError(
                    "Selected version not allowed " "Please chose an existing version"
                )
        np.random.seed()

        ##### END ######
        new_samples_final = np.zeros(
            (n_synthetic_sample, X_positifs_all_features.shape[1]), dtype=object
        )
        new_samples_final[:, bool_mask] = new_samples
        new_samples_final[:, ~bool_mask] = new_samples_cat

        X_positifs_final = np.zeros(
            (len(X_positifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_positifs_final[:, bool_mask] = X_positifs
        X_positifs_final[:, ~bool_mask] = X_positifs_categorical

        X_negatifs_final = np.zeros(
            (len(X_negatifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_negatifs_final[:, bool_mask] = X_negatifs
        X_negatifs_final[:, ~bool_mask] = X_negatifs_categorical

        oversampled_X = np.concatenate(
            (X_negatifs_final, X_positifs_final, new_samples_final), axis=0
        )
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y


class SMOTE_cat(BaseOverSampler):
    """
    SMOTE NC  distance-without-discrete-features
    """

    def __init__(
        self,
        K,
        categorical_features,
        version,
        sampling_strategy="auto",
        random_state=None,
    ):
        """
        K : int.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.categorical_features = categorical_features
        self.version = version
        self.random_state = random_state

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.categorical_features_.size == 0:
            raise ValueError(
                "SMOTE_cat is not designed to work only with numerical "
                "features. It requires some categorical features."
            )

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
        if len(self.categorical_features) == X.shape[1]:
            raise ValueError(
                "SMOTE_cat is not designed to work only with categorical "
                "features. It requires some numerical features."
            )

        bool_mask = np.ones((X_positifs.shape[1]), dtype=bool)
        bool_mask[self.categorical_features] = False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:, bool_mask]  ## continuous features
        X_negatifs = X_negatifs_all_features[:, bool_mask]  ## continuous features
        X_positifs_categorical = X_positifs_all_features[:, ~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:, ~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension = X_positifs.shape[1]  ## features continues seulement

        ######### CONTINUOUS ################
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbor_by_dist, neighbor_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=True
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, dimension))
        new_samples_cat = np.zeros(
            (n_synthetic_sample, len(self.categorical_features)), dtype=object
        )
        np.random.seed(self.random_state)
        for i in range(n_synthetic_sample):
            indice = np.random.randint(
                n_minoritaire
            )  # individu centrale qui sera choisi
            indice_neigh = np.random.randint(
                1, self.K + 1
            )  # Sélection voisin parmi les K. On exclu 0 car c indice lui-même
            indice_neighbor = neighbor_by_index[indice][indice_neigh]
            w = np.random.uniform(0, 1)  # facteur alpha de la première difference
            new_samples[i, :] = X_positifs[indice] + w * (
                X_positifs[indice_neighbor] - X_positifs[indice]
            )
            ############### CATEGORICAL ##################
            indice_neighbors_with_0 = np.arange(start=0, stop=self.K + 1, dtype=int)
            if (
                self.version == 1
            ):  ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    most_common = Counter(
                        X_positifs_categorical[indice_neighbors_with_0, cat_feature]
                    ).most_common(1)[0][0]
                    new_samples_cat[i, cat_feature] = most_common
            elif (
                self.version == 2
            ):  ## sampling of one of the nearest neighbors per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors_with_0, cat_feature],
                        replace=False,
                    )
            elif (
                self.version == 3
            ):  ## sampling of one of the nearest neighbors per categorical feature using dsitance
                #### We take the nn of the central point. The latter is excluded
                epsilon_weigths_sampling = 10e-6
                indice_neighbors_without_0 = np.arange(
                    start=1, stop=self.K + 1, dtype=int
                )
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors_without_0, cat_feature],
                        replace=False,
                        p=(
                            (
                                1
                                / (
                                    neighbor_by_dist[indice][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            )
                            / (
                                1
                                / (
                                    neighbor_by_dist[indice][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            ).sum()
                        ),
                    )
            else:
                raise ValueError(
                    "Selected version not allowed " "Please chose an existing version"
                )

        np.random.seed()

        ##### END ######
        new_samples_final = np.zeros(
            (n_synthetic_sample, X_positifs_all_features.shape[1]), dtype=object
        )
        new_samples_final[:, bool_mask] = new_samples
        new_samples_final[:, ~bool_mask] = new_samples_cat

        X_positifs_final = np.zeros(
            (len(X_positifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_positifs_final[:, bool_mask] = X_positifs
        X_positifs_final[:, ~bool_mask] = X_positifs_categorical

        X_negatifs_final = np.zeros(
            (len(X_negatifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_negatifs_final[:, bool_mask] = X_negatifs
        X_negatifs_final[:, ~bool_mask] = X_negatifs_categorical

        oversampled_X = np.concatenate(
            (X_negatifs_final, X_positifs_final, new_samples_final), axis=0
        )
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y

def proba_to_label(y_pred_probas, treshold=0.5):  # apply_threshold ?
    """_summary_

    Parameters
    ----------
    y_pred_probas : _type_
        _description_
    treshold : float, optional
        _description_, by default 0.5

    Returns
    -------
    _type_
        _description_
    """
    # Personnally I would do it in NumPy:
    return np.array(np.array(y_pred_probas) >= treshold, dtype=int)

from sklearn.covariance import ledoit_wolf,oas,empirical_covariance
class MultiOutPutClassifier_and_MGS(BaseOverSampler):
    """
    MultiOutPutClassifier and MGS
    """

    def __init__(
        self,
        K,
        categorical_features,
        Classifier,
        weighted_cov=False,
        ledoitwolfcov = False,
        oascov=False,
        tracecov=False,
        idcov=False,
        expcov=False,
        mucentered=False,
        to_encode=False,
        n_points=None,
        llambda=1.0,
        sampling_strategy="auto",
        random_state=None,
        bool_drf=False
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
        self.categorical_features = categorical_features
        self.Classifier = Classifier
        self.random_state = random_state
        self.to_encode = to_encode
        self.weighted_cov = weighted_cov
        self.ledoitwolfcov=ledoitwolfcov
        self.oascov=oascov
        self.tracecov=tracecov
        self.idcov=idcov
        self.mucentered=mucentered,
        self.expcov=expcov
        self.bool_drf=bool_drf
        

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        # X = _check_X(X)
        # self._check_n_features(X, reset=True)
        # self._check_feature_names(X, reset=True)
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.categorical_features_.size == 0:
            raise ValueError(
                "MultiOutPutClassifier_and_MGS is not designed to work only with numerical "
                "features. It requires some categorical features."
            )

    def fit_resample(self, X, y,scaler=None):  # Necessary only for SemiOracle
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """

        if scaler is None:
            output = self._fit_resample(X, y)
        else:
            output = self._fit_resample(X, y,scaler=scaler)

        X_,y_=output[0],output[1]
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    def _fit_resample(self, X, y=None,scaler=None, n_final_sample=None):
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
        if len(self.categorical_features) == X.shape[1]:
            raise ValueError(
                "MultiOutPutClassifier_and_MGS is not designed to work only with categorical "
                "features. It requires some numerical features."
            )
        bool_mask = np.ones((X_positifs.shape[1]), dtype=bool)
        bool_mask[self.categorical_features] = False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:, bool_mask]  ## continuous features
        X_negatifs = X_negatifs_all_features[:, bool_mask]  ## continuous features
        X_positifs_categorical = X_positifs_all_features[:, ~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:, ~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension = X_positifs.shape[1]  ## features continues seulement

        np.random.seed(self.random_state)
        if self.to_encode:
            ord_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, dtype=float
            )
            X_positifs_categorical_encoded = ord_encoder.fit_transform(
                X_positifs_categorical.astype(str)
            )
            ### Fit :
            self.Classifier.fit(
                X_positifs, X_positifs_categorical_encoded
            )  # learn on continuous features in order to predict categorical feature
        else:
            if self.bool_drf:
                self.Classifier.fit(
                    X_positifs, X_positifs_categorical
                )  # learn on continuous features in order to predict categorical features

            elif len(self.categorical_features)==1: # ravel in case of one categorical freatures
                self.Classifier.fit(
                    X_positifs, X_positifs_categorical.ravel().astype(str)
                )  # learn on continuous features in order to predict categorical features
            
            else:
                self.Classifier.fit(
                    X_positifs, X_positifs_categorical.astype(str)
                )  # learn on continuous features in order to predict categorical features
        ######### CONTINUOUS ################
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbors_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=False
        )
        n_synthetic_sample = n_final_sample - n_minoritaire
        if self.ledoitwolfcov:
            if self.mucentered:
                # We sample from central point
                mus = X_positifs
            else:
                # We sample from mean of neighbors
                all_neighbors = X_positifs[neighbors_by_index.flatten()]
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension
                    ).sum(axis=1)
            As = []
            for i in range(n_minoritaire):
                covariance, shrinkage = ledoit_wolf(X_positifs[neighbors_by_index[i,1:],:]-mus[neighbors_by_index[i,0]],assume_centered=True)
                As.append(self.llambda*covariance)
            As= np.array(As)   
        
        elif self.oascov:
            if self.mucentered:
                # We sample from central point
                mus = X_positifs
            else:
                # We sample from mean of neighbors
                all_neighbors = X_positifs[neighbors_by_index.flatten()]
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension
                    ).sum(axis=1)
            As = []
            for i in range(n_minoritaire):
                covariance, shrinkage = oas(X_positifs[neighbors_by_index[i,1:],:]-mus[neighbors_by_index[i,0]],assume_centered=True)
                As.append(self.llambda*covariance)
            As= np.array(As) 
        elif self.tracecov:
            if self.mucentered:
                # We sample from central point
                mus = X_positifs
            else:
                # We sample from mean of neighbors
                all_neighbors = X_positifs[neighbors_by_index.flatten()]
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension
                    ).sum(axis=1)
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                covariance  = empirical_covariance(X_positifs[neighbors_by_index[i,1:],:]-mus[neighbors_by_index[i,0]],assume_centered=True)
                final_covariance =  (np.trace(covariance)/p) * np.eye(p)
                As.append(self.llambda*final_covariance) 
            As= np.array(As) 
        elif self.idcov:
            if self.mucentered:
                # We sample from central point
                mus = X_positifs
            else:
                # We sample from mean of neighbors
                all_neighbors = X_positifs[neighbors_by_index.flatten()]
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension
                    ).sum(axis=1)
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                final_covariance =  (1/p) * np.eye(p)
                As.append(self.llambda*final_covariance) 
            As= np.array(As) 
        elif self.expcov:
            if self.mucentered:
                # We sample from central point
                mus = X_positifs
            else:
                # We sample from mean of neighbors
                all_neighbors = X_positifs[neighbors_by_index.flatten()]
                mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                    len(X_positifs), self.K + 1, dimension
                    ).sum(axis=1)
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                diffs = X_positifs[neighbors_by_index[i,1:],:]-mus[neighbors_by_index[i,0]]
                exp_dist = np.exp(-np.linalg.norm(diffs, axis=1))
                weights = exp_dist / (np.sum(exp_dist))
                final_covariance = (diffs.T.dot(np.diag(weights)).dot(diffs)) + np.eye(dimension) * 1e-10
                As.append(self.llambda* final_covariance) 
            As= np.array(As) 
            

        else:
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

            # spectral decomposition of all covariances
            # eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
            # eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
            # As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]
            As = np.linalg.cholesky(
                covs + 1e-10 * np.identity(dimension)
            )  ## add parameter for 1e-10 ?

        # sampling all new points
        # u = np.random.normal(loc=0, scale=1, size=(len(indices), dimension))
        # new_samples = [mus[central_point] + As[central_point].dot(u[central_point]) for i in indices]
        indices = np.random.randint(n_minoritaire, size=n_synthetic_sample)
        new_samples = np.zeros((n_synthetic_sample, dimension))
        for i, central_point in enumerate(indices):
            u = np.random.normal(loc=0, scale=1, size=dimension)
            new_observation = mus[central_point, :] + As[central_point].dot(u)
            new_samples[i, :] = new_observation
        ############### CATEGORICAL ##################
        if self.bool_drf: # special case of prediction for DRF
            out = self.Classifier.predict(newdata=new_samples, functional="weights")
            sample = np.zeros((new_samples.shape[0], out.y.shape[1]))
            for i in range(new_samples.shape[0]): 
                ids = np.random.choice(range(out.y.shape[0]), 1, p=out.weights[i, :])[0]
                sample[i,:] = out.y[ids,:]
            new_samples_cat = sample

        elif len(self.categorical_features)==1:# Ravel in case of one categorical freatures
            if scaler is None: # We give the scaler to the predictor
                new_samples_cat = self.Classifier.predict(new_samples).reshape(-1,1)
            else:
                new_samples_cat = self.Classifier.predict(new_samples,scaler=scaler).reshape(-1,1)
        else:
            if scaler is None:# We give the scaler to the predictor
                new_samples_cat = self.Classifier.predict(new_samples)  
            else :
                 new_samples_cat = self.Classifier.predict(new_samples,scaler=scaler)  
        np.random.seed()
        ##### END ######
        
        if self.to_encode:
            new_samples_cat = ord_encoder.inverse_transform(new_samples_cat.astype(int))
        new_samples_final = np.zeros(
            (n_synthetic_sample, X_positifs_all_features.shape[1]), dtype=object
        )
        new_samples_final[:, bool_mask] = new_samples
        new_samples_final[:, ~bool_mask] = new_samples_cat
        # new_samples_final = np.concatenate((new_samples,new_samples_cat), axis=1)

        X_positifs_final = np.zeros(
            (len(X_positifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_positifs_final[:, bool_mask] = X_positifs
        X_positifs_final[:, ~bool_mask] = X_positifs_categorical

        X_negatifs_final = np.zeros(
            (len(X_negatifs), X_positifs_all_features.shape[1]), dtype=object
        )
        X_negatifs_final[:, bool_mask] = X_negatifs
        X_negatifs_final[:, ~bool_mask] = X_negatifs_categorical

        oversampled_X = np.concatenate(
            (X_negatifs_final, X_positifs_final, new_samples_final), axis=0
        )
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y
    

class OracleOverSampler(BaseOverSampler):
    """
    MGS : Multivariate Gaussian SMOTE
    """

    def __init__(
        self, generator,generator_params=None,sampling_strategy="auto", random_state=None, to_str=True
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.generator=generator
        self.generator_params=generator_params
        self.to_str=to_str

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        # X = _check_X(X)
        # self._check_n_features(X, reset=True)
        # self._check_feature_names(X, reset=True)
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.categorical_features_.size == 0:
            raise ValueError(
                "MultiOutPutClassifier_and_MGS is not designed to work only with numerical "
                "features. It requires some categorical features."
            )

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
        dimension = X_positifs.shape[1]
        n_synthetic_sample = n_final_sample - n_minoritaire
        if self.to_str:
            new_samples = np.zeros((n_synthetic_sample, dimension),dtype=str) # general case
        else: 
            new_samples = np.zeros((n_synthetic_sample, dimension)) # case only  binary categorical features
        current_all_n=0
        seed_generator=self.random_state 
        while current_all_n < n_synthetic_sample :
            if self.generator_params is None:
                curr_X,curr_target,curr_target_num = self.generator(n_samples=n_synthetic_sample,dimension=dimension,random_state=seed_generator,verbose=-1)
            else:
                curr_X,curr_target,curr_target_num = self.generator(n_samples=n_synthetic_sample,dimension=dimension,random_state=seed_generator,verbose=-1,**self.generator_params)
                

            curr_minority_X = curr_X[curr_target_num==1]
            if current_all_n+len(curr_minority_X) < n_synthetic_sample:
                new_samples[current_all_n:current_all_n+len(curr_minority_X),:] = curr_minority_X
                current_all_n+=len(curr_minority_X)
            else:
                new_samples[current_all_n:,:] = curr_minority_X[:(n_synthetic_sample-current_all_n),:]
                current_all_n = n_synthetic_sample
            seed_generator+=1


        oversampled_X = np.concatenate((X_negatifs, X_positifs, new_samples), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )
        
        return oversampled_X, oversampled_y
        

from data.data import generate_synthetic_features_logreg
class OracleOneCat():
    def __init__(self):
        pass
    def fit(self,X,y):
        pass
    def predict(self,X,scaler,y=None):
        inversed_X = scaler.inverse_transform(X)
        feature_cat_uniform, feature_cat_uniform_numeric = generate_synthetic_features_logreg(X=inversed_X,index_informatives=[0,1,2],list_modalities=['C','D'],beta=np.array([-8,7,6]),intercept=-2)
        return feature_cat_uniform_numeric
    
from data.data import generate_synthetic_features_logreg_triple
class OracleTwoCat():
    def __init__(self):
        pass
    def fit(self,X,y):
        pass
    def predict(self,X,scaler,y=None):
        inversed_X = scaler.inverse_transform(X)
        
        feature_cat_uniform, feature_cat_uniform_numeric = generate_synthetic_features_logreg_triple(X=inversed_X,index_informatives=[0,1,2],
                                                                                list_modalities=['A','B','C'],
                                                                                beta1=np.array([-8,7,6]),beta2=np.array([4,-7,3]),beta3=np.array([2,-1,2])
                                                                                            )

        feature_cat_uniform2, feature_cat_uniform_numeric2 = generate_synthetic_features_logreg_triple(X=inversed_X,index_informatives=[0,1,2],
                                                                                    list_modalities=['D','E','F'],
                                                                                    beta1=np.array([-4,5,6]),beta2=np.array([6,-3,2]),beta3=np.array([1,5,-1])
                                                                                                )
        feature_cat_uniform_numeric_final = np.hstack((feature_cat_uniform_numeric.reshape(-1,1),feature_cat_uniform_numeric2.reshape(-1,1)))
        return feature_cat_uniform_numeric_final


# SMOTE ENC from authors :
from sklearn.utils import (
    check_array,
    _safe_indexing,
    sparsefuncs_fast,
    check_random_state,
)

# from scipy import stats
from numbers import Integral
from scipy import sparse
import pandas as pd
from sklearn import clone
from sklearn.neighbors._base import KNeighborsMixin
from imblearn.exceptions import raise_isinstance_error


class SMOTE_ENC:
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features

    def chk_neighbors(self, nn_object, additional_neighbor):
        if isinstance(nn_object, Integral):
            return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
        elif isinstance(nn_object, KNeighborsMixin):
            return clone(nn_object)
        else:
            raise_isinstance_error(
                nn_name, [int, KNeighborsMixin], nn_object
            )  ### A regarder en détail

    def generate_samples(
        self,
        X,
        nn_data,
        nn_num,
        rows,
        cols,
        steps,
        continuous_features_,
    ):
        rng = check_random_state(42)

        diffs = nn_data[nn_num[rows, cols]] - X[rows]

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs

        X_new = X_new.tolil() if sparse.issparse(X_new) else X_new
        # convert to dense array since scipy.sparse doesn't handle 3D
        nn_data = nn_data.toarray() if sparse.issparse(nn_data) else nn_data

        all_neighbors = nn_data[nn_num[rows]]

        for idx in range(continuous_features_.size, X.shape[1]):
            mode = stats.mode(all_neighbors[:, :, idx], axis=1)[0]

            X_new[:, idx] = np.ravel(mode)
        return X_new

    def make_samples(
        self,
        X,
        y_dtype,
        y_type,
        nn_data,
        nn_num,
        n_samples,
        continuous_features_,
        step_size=1.0,
    ):
        random_state = check_random_state(42)
        samples_indices = random_state.randint(
            low=0, high=len(nn_num.flatten()), size=n_samples
        )
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self.generate_samples(
            X, nn_data, nn_num, rows, cols, steps, continuous_features_
        )
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)

        return X_new, y_new

    def cat_corr_pandas(self, X, target_df, target_column, target_value):
        # X has categorical columns
        categorical_columns = list(X.columns)
        X = pd.concat([X, target_df], axis=1)

        # filter X for target value
        is_target = X.loc[:, target_column] == target_value
        X_filtered = X.loc[is_target, :]

        X_filtered.drop(target_column, axis=1, inplace=True)

        # get columns in X
        nrows = len(X)
        encoded_dict_list = []
        nan_dict = dict({})
        c = 0
        imb_ratio = len(X_filtered) / len(X)
        OE_dict = {}

        for column in categorical_columns:
            for level in list(X.loc[:, column].unique()):
                # filter rows where level is present
                row_level_filter = X.loc[:, column] == level
                rows_in_level = len(X.loc[row_level_filter, :])

                # number of rows in level where target is 1
                O = len(X.loc[is_target & row_level_filter, :])
                E = rows_in_level * imb_ratio
                # Encoded value = chi, i.e. (observed - expected)/expected
                ENC = (O - E) / E
                OE_dict[level] = ENC

            encoded_dict_list.append(OE_dict)

            X.loc[:, column] = X[column].map(OE_dict)
            nan_idx_array = np.ravel(
                np.argwhere(np.isnan(X.loc[:, column]).to_numpy())
            )  ## Add .tonumpy() Abd
            if len(nan_idx_array) > 0:
                nan_dict[c] = nan_idx_array
            c = c + 1
            X.loc[:, column].fillna(-1, inplace=True)

        X.drop(target_column, axis=1, inplace=True)
        return X, encoded_dict_list, nan_dict

    def fit_resample(self, X, y):
        X = pd.DataFrame(X)  ## ABD
        y = pd.DataFrame({"target": y})  ##ABD
        X_cat_encoded, encoded_dict_list, nan_dict = self.cat_corr_pandas(
            X.iloc[:, np.asarray(self.categorical_features)],
            y,
            target_column="target",
            target_value=1,
        )
        X_cat_encoded = np.array(X_cat_encoded)
        y = np.ravel(y)
        X = np.array(X)

        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {
            key: n_sample_majority - value
            for (key, value) in target_stats.items()
            if key != class_majority
        }

        n_features_ = X.shape[1]
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == "bool":
            categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any([cat not in np.arange(n_features_) for cat in categorical_features]):
                raise ValueError(
                    "Some of the categorical indices are out of range. Indices"
                    " should be between 0 and {}".format(n_features_)
                )
            categorical_features_ = categorical_features

        continuous_features_ = np.setdiff1d(
            np.arange(n_features_), categorical_features_
        )

        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=["csr", "csc"])
        X_minority = _safe_indexing(X_continuous, np.flatnonzero(y == class_minority))

        if sparse.issparse(X):
            if X.format == "csr":
                _, var = sparsefuncs_fast.csr_mean_variance_axis0(X_minority)
            else:
                _, var = sparsefuncs_fast.csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        median_std_ = np.median(np.sqrt(var))

        X_categorical = X[:, categorical_features_]
        X_copy = np.hstack((X_continuous, X_categorical))

        X_cat_encoded = X_cat_encoded * median_std_

        X_encoded = np.hstack((X_continuous, X_cat_encoded))
        #X_resampled = X_encoded.copy() # ABD. In order to have initial data not modified
        X_resampled  = X_copy.copy()#ABD. In order to have initial data not modified
        y_resampled = y.copy()

        for class_sample, n_samples in sampling_strategy.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X_encoded, target_class_indices)
            nn_k_ = self.chk_neighbors(5, 1)
            nn_k_.fit(X_class)

            nns = nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self.make_samples(
                X_class,
                y.dtype,
                class_sample,
                X_class,
                nns,
                n_samples,
                continuous_features_,
                1.0,
            )
            i = 0 ## ABD. In order to have initial data not modified
            #print('Inside SMOTE NC X_new : ',X_new) 
            for col in range(continuous_features_.size, X.shape[1]):
                encoded_dict = encoded_dict_list[i]
                i = i + 1
                for key, value in encoded_dict.items():
                    X_new[:, col] = np.where(
                       np.round(X_new[:, col], 4)
                        == np.round(value * median_std_, 4),
                        key,
                        X_new[:, col],
                    )## END ABD.

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
                sparse_func = "tocsc" if X.format == "csc" else "tocsr"
                X_resampled = getattr(X_resampled, sparse_func)()
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        X_resampled_copy = X_resampled.copy()
        #i = 0 ## ABD. In order to have initial data not modified
        #print('Inside SMOTE NC X_resampled_copy : ',X_resampled_copy) ## ABD
        #for col in range(continuous_features_.size, X.shape[1]):
        #   encoded_dict = encoded_dict_list[i]
        #   i = i + 1
        #   for key, value in encoded_dict.items():
        #        X_resampled_copy[:, col] = np.where(
        #            np.round(X_resampled_copy[:, col], 4)
        #            == np.round(value * median_std_, 4),
        #            key,
        #            X_resampled_copy[:, col],
        #        )
        ## END ABD

        for key, value in nan_dict.items():
            for item in value:
                X_resampled_copy[item, continuous_features_.size + key] = X_copy[
                    item, continuous_features_.size + key
                ]

        X_resampled = X_resampled_copy
        indices_reordered = np.argsort(
            np.hstack((continuous_features_, categorical_features_))
        )
        if sparse.issparse(X_resampled):
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]
        return X_resampled, y_resampled


from sklearn.preprocessing import OrdinalEncoder


class SMOTE_ENC_decoded(SMOTE_ENC):
    def __init__(self, categorical_features):
        super().__init__(categorical_features)

    def fit_resample(self, X, y):
        ord_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=int
        )
        X_copy = X.copy()
        X_copy[:, self.categorical_features] = ord_encoder.fit_transform(
            X_copy[:, self.categorical_features]
        )
        print('X after encoding : ',X_copy)
        print('X shape after encoding : ',X_copy.shape)
        ### Sampling :
        X_res, y_res = super().fit_resample(X_copy, y)
        print('X_res after sampling : ',X_res)
        print('X_res shape after encoding : ',X_res.shape)
        print('X_rescategorical  shape after encoding : ',X_res[:, self.categorical_features])
        #print('Dtype :',X_res.dtype)
        X_res =  X_res.astype(str)
        #print('Dtype after str :',type(X_res))
        X_res[:, self.categorical_features] = ord_encoder.inverse_transform(
            X_res[:, self.categorical_features].astype(float)
        )
        return X_res, y_res
    


from sklearn.ensemble._forest import ForestClassifier
class DrfSk(RandomForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super(ForestClassifier,self).__init__(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.monotonic_cst = monotonic_cst
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        self.trained_X = X
        self.trained_y = y
        super().fit(X=X,y=y,sample_weight=sample_weight)

    def get_weights(self,x):   
        n_tree = len(clf.estimators_)
        w=np.zeros((len(self.trained_X),))
        for tree in self.estimators_:
            train_samples_leaves = tree.apply(self.trained_X)
            x_leaf = tree.apply(x)[0]
            indices_train_samples_in_same_leaf = np.where(train_samples_leaves==x_leaf)[0]
            n_leaves_in = len(indices_train_samples_in_same_leaf)
            if n_leaves_in != 0:
                for idx in indices_train_samples_in_same_leaf:
                    w[idx] = w[idx] + 1/(n_tree*n_leaves_in)
        return w

    def predict(self,X):
        #weights_all = np.apply_along_axis(self.get_weights,0,X,{'self':self})
        size_train = len(self.trained_X)
        list_index_train_X = np.arange(start=0,stop=size_train,step=1)
        y_pred = []
        for i in range(len(X)):
            x = X[i,:].reshape(1, -1)
            w = self.get_weights(x)
            selected_index = np.random.choice(a=list_index_train_X,size=1,replace=False,p=w)
            y_pred.append(self.trained_y[selected_index])
            
        return np.array(y_pred).reshape(-1,)
