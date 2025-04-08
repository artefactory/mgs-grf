import math
from collections import namedtuple  ## KNN

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_target_type
from sklearn.covariance import empirical_covariance, ledoit_wolf, oas
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


class NoSampling:
    """
    None rebalancing strategy class
    """

    def fit_resample(self, X, y):
        """
        X is a numpy array
        y is a numpy array of dimension (n,)
        """
        return X, y


ModeResult = namedtuple("ModeResult", ("mode", "count"))


def mode_rand(a, axis):
    in_dims = list(range(a.ndim))
    a_view = np.transpose(a, in_dims[:axis] + in_dims[axis + 1 :] + [axis])

    inds = np.ndindex(a_view.shape[:-1])
    modes = np.empty(a_view.shape[:-1], dtype=a.dtype)
    counts = np.zeros(a_view.shape[:-1], dtype=int)

    for ind in inds:
        vals, cnts = np.unique(a_view[ind], return_counts=True)
        maxes = np.where(cnts == cnts.max())  # Here's the change
        modes[ind], counts[ind] = vals[np.random.choice(maxes[0])], cnts.max()

    newshape = list(a.shape)
    newshape[axis] = 1
    return ModeResult(modes.reshape(newshape), counts.reshape(newshape))


##########################################
######## CATEGORICAL #####################
#########################################
class WMGS_NC_cov(BaseOverSampler):
    """
    MGS NC strategy
    """

    def __init__(
        self,
        K,
        categorical_features,
        version,
        kind_sampling="cholesky",
        kind_cov="Emp",
        mucentered=True,
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
        self.kind_sampling = kind_sampling
        self.kind_cov = kind_cov
        self.mucentered = mucentered
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

    def fit_resample(self, X, y):
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

        output = self._fit_resample(X, y)
        X_, y_ = output[0], output[1]
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    def _fit_resample(self, X, y=None, n_final_sample=None):
        """
        if y=None, all points are considered positive, and oversampling on all X
        if n_final_sample=None, objective is balanced data.
        """

        if y is None:
            X_positifs = X
            X_negatifs = np.ones((0, X.shape[1]))
            assert n_final_sample is not None, "You need to provide a number of final samples."
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
        X_positifs_categorical_enc = enc.fit_transform(X_positifs_categorical).toarray()
        X_positifs_all_features_enc = np.hstack((X_positifs, X_positifs_categorical_enc))
        cste_med = np.median(
            np.sqrt(np.var(X_positifs, axis=0))
        )  ## med constante from continuous variables
        if not math.isclose(cste_med, 0):
            X_positifs_all_features_enc[:, dimension_continuous:] = X_positifs_all_features_enc[
                :, dimension_continuous:
            ] * (cste_med / np.sqrt(2))
            # With one-hot encoding, the median will be repeated twice. We need
        # to divide by sqrt(2) such that we only have one median value
        # contributing to the Euclidean distance
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs_all_features_enc)
        neighbor_by_dist, neighbor_by_index = neigh.kneighbors(
            X=X_positifs_all_features_enc, n_neighbors=self.K + 1, return_distance=True
        )
        n_synthetic_sample = n_final_sample - n_minoritaire
        np.random.seed(self.random_state)

        if self.mucentered:
            # We sample from mean of neighbors
            all_neighbors = X_positifs[neighbor_by_index.flatten()]
            mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                len(X_positifs), self.K + 1, dimension_continuous
            ).sum(axis=1)
        else:
            # We sample from central point
            mus = X_positifs

        if self.kind_cov == "EmpCov" or self.kind_cov == "InvWeightCov":
            centered_X = X_positifs[neighbor_by_index.flatten()] - np.repeat(
                mus, self.K + 1, axis=0
            )
            centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension_continuous)
            if self.kind_cov == "InvWeightCov":
                distances = (centered_X**2).sum(axis=-1)
                distances[distances > 1e-10] = distances[distances > 1e-10] ** -0.25

                # inv sqrt for positives only and half of power for multiplication below
                distances /= distances.sum(axis=-1)[:, np.newaxis]
                centered_X = (
                    np.repeat(distances[:, :, np.newaxis] ** 0.5, dimension_continuous, axis=2)
                    * centered_X
                )

            covs = (
                self.llambda * np.matmul(np.swapaxes(centered_X, 1, 2), centered_X) / (self.K + 1)
            )
            if self.kind_sampling == "svd":
                # spectral decomposition of all covariances
                eigen_values, eigen_vectors = np.linalg.eigh(covs)  ## long
                eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** 0.5
                As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]
            elif self.kind_sampling == "cholesky":
                As = np.linalg.cholesky(covs + 1e-10 * np.identity(dimension_continuous))
            else:
                raise ValueError(
                    "kind_sampling of MGS not supportedAvailable values : 'cholescky','svd' "
                )

        elif self.kind_cov == "LWCov":
            As = []
            for i in range(n_minoritaire):
                covariance, shrinkage = ledoit_wolf(
                    X_positifs[neighbor_by_index[i, 1:], :] - mus[neighbor_by_index[i, 0]],
                    assume_centered=True,
                )
                As.append(self.llambda * covariance)
            As = np.array(As)

        elif self.kind_cov == "OASCov":
            As = []
            for i in range(n_minoritaire):
                covariance, shrinkage = oas(
                    X_positifs[neighbor_by_index[i, 1:], :] - mus[neighbor_by_index[i, 0]],
                    assume_centered=True,
                )
                As.append(self.llambda * covariance)
            As = np.array(As)
        elif self.kind_cov == "TraceCov":
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                covariance = empirical_covariance(
                    X_positifs[neighbor_by_index[i, 1:], :] - mus[neighbor_by_index[i, 0]],
                    assume_centered=True,
                )
                final_covariance = (np.trace(covariance) / p) * np.eye(p)
                As.append(self.llambda * final_covariance)
            As = np.array(As)
        elif self.kind_cov == "IdCov":
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                final_covariance = (1 / p) * np.eye(p)
                As.append(self.llambda * final_covariance)
            As = np.array(As)
        elif self.kind_cov == "ExpCov":
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                diffs = X_positifs[neighbor_by_index[i, 1:], :] - mus[neighbor_by_index[i, 0]]
                exp_dist = np.exp(-np.linalg.norm(diffs, axis=1))
                weights = exp_dist / (np.sum(exp_dist))
                final_covariance = (diffs.T.dot(np.diag(weights)).dot(diffs)) + np.eye(
                    dimension_continuous
                ) * 1e-10
                As.append(self.llambda * final_covariance)
            As = np.array(As)

        else:
            raise ValueError(
                "kind_cov of MGS not supported"
                "Available values : 'EmpCov','InvWeightCov','LWCov','OASCov','TraceCov','IdCov','ExpCov' "
            )

        # sampling all new points
        # u = np.random.normal(loc=0, scale=1, size=(len(indices), dimension))
        # new_samples = [mus[central_point] + As[central_point].dot(u[central_point]) for i in indices]
        indices = np.random.randint(n_minoritaire, size=n_synthetic_sample)
        new_samples = np.zeros((n_synthetic_sample, dimension_continuous))
        new_samples_cat = np.zeros(
            (n_synthetic_sample, len(self.categorical_features)), dtype=object
        )
        for i, central_point in enumerate(indices):
            u = np.random.normal(loc=0, scale=1, size=dimension_continuous)
            new_observation = mus[central_point, :] + As[central_point].dot(u)
            new_samples[i, :] = new_observation
            ############### CATEGORICAL ##################
            indices_neigh = np.arange(1, self.K + 1, 1)
            indice_neighbors = neighbor_by_index[central_point][indices_neigh]

            if self.version == 1:  ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    most_common, _ = mode_rand(
                        X_positifs_categorical[indice_neighbors, cat_feature].ravel(),
                        axis=0,
                    )
                    new_samples_cat[i, cat_feature] = most_common[0]
                    # vals, cnts = np.unique(X_positifs_categorical[indice_neighbors, cat_feature].ravel(), return_counts=True)
                    # ind_maxes = np.flatnonzero(cnts == np.max(cnts))
                    # selected_ind = np.random.choice(ind_maxes)
                    # new_samples_cat[i, cat_feature] = vals[selected_ind]
            elif (
                self.version == 2
            ):  ## sampling of one of the nearest neighbors per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    selected_one = np.random.choice(
                        X_positifs_categorical[indice_neighbors, cat_feature],
                        replace=False,
                    )
                    new_samples_cat[i, cat_feature] = selected_one
            elif (
                self.version == 3
            ):  ## sampling of one of the nearest neighbors per categorical feature using dsitance
                #### We take the nn of the central point. The latter is excluded
                print("Version 3")
                epsilon_weigths_sampling = 10e-6
                indice_neighbors_without_0 = np.arange(start=1, stop=self.K + 1, dtype=int)
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(
                        X_positifs_categorical[indice_neighbors_without_0, cat_feature],
                        replace=False,
                        p=(
                            (
                                1
                                / (
                                    neighbor_by_dist[central_point][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            )
                            / (
                                1
                                / (
                                    neighbor_by_dist[central_point][indice_neighbors_without_0]
                                    + epsilon_weigths_sampling
                                )
                            ).sum()
                        ),
                    )
            else:
                raise ValueError("Selected version not allowed Please chose an existing version")

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
        oversampled_y = np.hstack((np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1)))
        np.random.seed()

        return oversampled_X, oversampled_y
