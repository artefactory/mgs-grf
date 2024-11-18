
import random
import math
from typing import Optional, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE

from imblearn.over_sampling.base import BaseOverSampler
from sklearn.neighbors import NearestNeighbors


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