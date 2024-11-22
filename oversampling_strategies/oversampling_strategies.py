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
class WMGS_NC_cov(BaseOverSampler):
    """
    MGS NC strategy
    """

    def __init__(
        self,
        K,
        categorical_features,
        version,
        kind_sampling='cholesky',
        kind_cov = 'Emp',
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
        X_,y_=output[0],output[1]
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

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

        if self.kind_cov=='EmpCov' or self.kind_cov=='InvWeightCov':
            centered_X = X_positifs[neighbor_by_index.flatten()] - np.repeat(
                mus, self.K + 1, axis=0
            )
            centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension_continuous)
            if self.kind_cov=='InvWeightCov':
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
            if self.kind_sampling == 'svd':
            # spectral decomposition of all covariances
                eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
                eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
                As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]
            elif self.kind_sampling == 'cholesky' :
                As = np.linalg.cholesky(
                    covs + 1e-10 * np.identity(dimension_continuous)
                ) 
            else: 
                raise ValueError(
                        "kind_sampling of MGS not supported"
                        "Available values : 'cholescky','svd' "
                    )

        elif self.kind_cov=='LWCov':
            As = []
            for i in range(n_minoritaire):
                covariance, shrinkage = ledoit_wolf(X_positifs[neighbor_by_index[i,1:],:]-mus[neighbor_by_index[i,0]],assume_centered=True)
                As.append(self.llambda*covariance)
            As= np.array(As)   
        
        elif self.kind_cov=='OASCov':
            As = []
            for i in range(n_minoritaire):
                covariance, shrinkage = oas(X_positifs[neighbor_by_index[i,1:],:]-mus[neighbor_by_index[i,0]],assume_centered=True)
                As.append(self.llambda*covariance)
            As= np.array(As) 
        elif self.kind_cov=='TraceCov':
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                covariance  = empirical_covariance(X_positifs[neighbor_by_index[i,1:],:]-mus[neighbor_by_index[i,0]],assume_centered=True)
                final_covariance = (np.trace(covariance)/p) * np.eye(p)
                As.append(self.llambda*final_covariance) 
            As= np.array(As) 
        elif self.kind_cov=='IdCov':
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                final_covariance = (1/p) * np.eye(p)
                As.append(self.llambda*final_covariance) 
            As= np.array(As) 
        elif self.kind_cov=='ExpCov':
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
        #for i in range(n_synthetic_sample):
            #indice = central_point
            indices_neigh = []## the central point is NOT selected for the construction of the categorical features (votes)
            indices_neigh.extend(
                np.random.choice(a=range(1, self.K + 1),size=self.n_points,replace=False)
            )  # The nearrest neighbor selected for the estimation
            indice_neighbors = neighbor_by_index[central_point][indices_neigh]

            if (
                self.version == 1
            ):  ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    list_neigh_value = np.random.permutation(X_positifs_categorical[indice_neighbors, cat_feature]) ## We randomly permute because Counter select the first seen in case of tie
                    most_common = Counter(
                        list_neigh_value
                    ).most_common(1)[0][0]
                    new_samples_cat[i, cat_feature] = most_common
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
                raise ValueError(
                    "Selected version not allowed " "Please chose an existing version"
                )
        

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
        np.random.seed()

        return oversampled_X, oversampled_y
    

class MultiOutPutClassifier_and_MGS(BaseOverSampler):
    """
    MultiOutPutClassifier and MGS
    """

    def __init__(
        self,
        K,
        categorical_features,
        Classifier,
        kind_sampling='cholesky',
        kind_cov = 'EmpCov',
        mucentered=True,
        to_encode=False,
        to_encode_onehot=False,
        n_points=None,
        llambda=1.0,
        sampling_strategy="auto",
        random_state=None,
        bool_drf=False,
        bool_rf=False,
        bool_rf_regressor=False,
        bool_drfsk_regressor=False,
        fit_nn_on_continuous_only=True,
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
        self.kind_cov=kind_cov
        self.kind_sampling=kind_sampling
        self.mucentered=mucentered
        self.random_state = random_state
        self.to_encode = to_encode ## encode categorical Z vector with ordinal encoding
        self.to_encode_onehot = to_encode_onehot ## encode categorical Z vector with one hot encoding
        self.bool_rf = bool_rf ## Perform special predictt of RFClassifier when to_encode_onehot=True
        self.bool_rf_regressor =bool_rf_regressor ##Perform special predictt of RFRegressor in when to_encode_onehot=True
        self.bool_drfsk_regressor = bool_drfsk_regressor
        self.bool_drf=bool_drf
        self.fit_nn_on_continuous_only = fit_nn_on_continuous_only
        

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

    def fit_resample(self, X, y,scaler=None):  # scaler Necessary only for SemiOracle
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
        dimension_continuous = X_positifs.shape[1]  ## features continues seulement

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
        elif self.to_encode_onehot:
            onehot_encoder = OneHotEncoder(handle_unknown='ignore',dtype=float,sparse_output=False)
            X_positifs_categorical_encoded = onehot_encoder.fit_transform(
                X_positifs_categorical.astype(str)
            )
            if self.bool_rf_regressor or self.bool_drfsk_regressor:  ## When using regressorsn the data are scaled. Because regressor predict is tretaed diffrently (probas got diffrently)
                var_scaler_cat = StandardScaler(with_mean=False,with_std=True)
                X_positifs_categorical_encoded = var_scaler_cat.fit_transform(X_positifs_categorical_encoded) ## we scale the categorical variables 
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
        
        if self.fit_nn_on_continuous_only: # We fit the nn estimator only on the continuous features
            neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
            neigh.fit(X_positifs)
            neighbor_by_index = neigh.kneighbors(
                X=X_positifs, n_neighbors=self.K + 1, return_distance=False
            )
        else: # We fit the nn estimator on the continuous features and add the mean (NC like).
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
            neighbor_by_index = neigh.kneighbors(
                X=X_positifs_all_features_enc, n_neighbors=self.K + 1, return_distance=False
            )


        n_synthetic_sample = n_final_sample - n_minoritaire
        if self.mucentered:
            # We sample from mean of neighbors
            all_neighbors = X_positifs[neighbor_by_index.flatten()]
            mus = (1 / (self.K + 1)) * all_neighbors.reshape(
                len(X_positifs), self.K + 1, dimension_continuous
                ).sum(axis=1)
        else:
                # We sample from central point
            mus = X_positifs

        if self.kind_cov=='EmpCov' or self.kind_cov=='InvWeightCov':
            centered_X = X_positifs[neighbor_by_index.flatten()] - np.repeat(
                mus, self.K + 1, axis=0
            )
            centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension_continuous)
            if self.kind_cov=='InvWeightCov':
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
            if self.kind_sampling == 'svd':
            # spectral decomposition of all covariances
                eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
                eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
                As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]
            elif self.kind_sampling == 'cholesky' :
                As = np.linalg.cholesky(
                    covs + 1e-10 * np.identity(dimension_continuous)
                ) 
            else: 
                raise ValueError(
                        "kind_sampling of MGS not supported"
                        "Available values : 'cholescky','svd' "
                    )

        elif self.kind_cov=='LWCov':
            As = []
            for i in range(n_minoritaire):
                covariance, shrinkage = ledoit_wolf(X_positifs[neighbor_by_index[i,1:],:]-mus[neighbor_by_index[i,0]],assume_centered=True)
                As.append(self.llambda*covariance)
            As= np.array(As)   
        
        elif self.kind_cov=='OASCov':
            As = []
            for i in range(n_minoritaire):
                covariance, shrinkage = oas(X_positifs[neighbor_by_index[i,1:],:]-mus[neighbor_by_index[i,0]],assume_centered=True)
                As.append(self.llambda*covariance)
            As= np.array(As) 
        elif self.kind_cov=='TraceCov':
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                covariance  = empirical_covariance(X_positifs[neighbor_by_index[i,1:],:]-mus[neighbor_by_index[i,0]],assume_centered=True)
                final_covariance = (np.trace(covariance)/p) * np.eye(p)
                As.append(self.llambda*final_covariance) 
            As= np.array(As) 
        elif self.kind_cov=='IdCov':
            As = []
            p = X_positifs.shape[1]
            for i in range(n_minoritaire):
                final_covariance = (1/p) * np.eye(p)
                As.append(self.llambda*final_covariance) 
            As= np.array(As) 
        elif self.kind_cov=='ExpCov':
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
            raise ValueError(
                        "kind_cov of MGS not supported"
                        "Available values : 'EmpCov','InvWeightCov','LWCov','OASCov','TraceCov','IdCov','ExpCov' "
                    )
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
        if self.bool_drf: # special case of prediction for DRF (from the original article)
            out = self.Classifier.predict(newdata=new_samples, functional="weights")
            sample = np.zeros((new_samples.shape[0], out.y.shape[1]))
            for i in range(new_samples.shape[0]): 
                ids = np.random.choice(range(out.y.shape[0]), 1, p=out.weights[i, :])[0]
                sample[i,:] = out.y[ids,:]
            new_samples_cat = sample

        if self.bool_rf and self.to_encode_onehot:
            categories_onehot = onehot_encoder.categories_
            new_samples_cat_probas_all = self.Classifier.predict_proba(new_samples) # list of pred_probas for each categorical one hot encoded.
            new_samples_cat_probas  = new_samples_cat_probas_all[0][:,1].reshape(-1,1)
            for i in range(1,len(onehot_encoder.get_feature_names_out())):
                new_samples_cat_probas = np.hstack((new_samples_cat_probas,new_samples_cat_probas_all[i][:,1].reshape(-1,1)))
            new_samples_cat = np.zeros((new_samples_cat_probas.shape[0],new_samples_cat_probas.shape[1]))
            start_idx = 0
            for i in range(len(categories_onehot)):
                curr_n_modalities = len(categories_onehot[i])
                indices_argmax = np.argmax(new_samples_cat_probas[:,start_idx:(start_idx+curr_n_modalities)],axis=1) 
                indices_argmax = start_idx + indices_argmax
                new_samples_cat[np.arange(len(new_samples_cat)),indices_argmax] = 1
                start_idx = curr_n_modalities
        elif (self.bool_rf_regressor  or self.bool_drfsk_regressor) and self.to_encode_onehot:
            categories_onehot = onehot_encoder.categories_
            new_samples_cat_pred= self.Classifier.predict(new_samples) # list of  pred for each categorical one hot encoded.
            new_samples_cat_pred_probas = var_scaler_cat.inverse_transform(new_samples_cat_pred)
            new_samples_cat = np.zeros((new_samples_cat_pred_probas.shape[0],new_samples_cat_pred_probas.shape[1]))
            start_idx = 0
            for i in range(len(categories_onehot)):
                curr_n_modalities = len(categories_onehot[i])
                indices_argmax = np.argmax(new_samples_cat_pred_probas[:,start_idx:(start_idx+curr_n_modalities)],axis=1) 
                indices_argmax = start_idx + indices_argmax
                new_samples_cat[np.arange(len(new_samples_cat)),indices_argmax] = 1
                start_idx = curr_n_modalities
        elif len(self.categorical_features)==1:# Ravel in case of one categorical freatures
            if scaler is None: 
                new_samples_cat = self.Classifier.predict(new_samples).reshape(-1,1)
            else:# We give the scaler to the predictor (for SemiOracle )
                new_samples_cat = self.Classifier.predict(new_samples,scaler=scaler).reshape(-1,1)
        else:
            if scaler is None:
                new_samples_cat = self.Classifier.predict(new_samples)  
            else :# We give the scaler to the predictor (for SemiOracle )
                 new_samples_cat = self.Classifier.predict(new_samples,scaler=scaler)  
        np.random.seed()
        ##### END ######
        
        if self.to_encode:
            new_samples_cat = ord_encoder.inverse_transform(new_samples_cat.astype(int))
        elif self.to_encode_onehot:
            new_samples_cat = onehot_encoder.inverse_transform(new_samples_cat.astype(int))
            
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
    


from sklearn.ensemble._forest import ForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
        self.train_samples_leaves = []
        for tree in self.estimators_:
            self.train_samples_leaves.append(tree.apply(self.trained_X)) 

    def get_weights(self,x):   
        n_tree = len(self.estimators_)
        w=np.zeros((len(self.trained_X),))
        for t,tree in enumerate(self.estimators_):
            #train_samples_leaves = tree.apply(self.trained_X)
            train_samples_leaves = self.train_samples_leaves[t]
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
        y_pred = np.zeros((len(X),self.trained_y.shape[1]))
        for i in range(len(X)):
            x = X[i,:].reshape(1, -1)
            w = self.get_weights(x)
            selected_index = np.random.choice(a=list_index_train_X,size=1,replace=False,p=w)
            y_pred[i] = self.trained_y[selected_index]
            
        return np.array(y_pred)
    


from sklearn.ensemble._forest import ForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
class DrfSkRegressor(RandomForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super(ForestRegressor,self).__init__(
            estimator=DecisionTreeRegressor(),
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
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

    def fit(self, X, y, sample_weight=None):
        self.trained_X = X
        self.trained_y = y
        super().fit(X=X,y=y,sample_weight=sample_weight)
        self.train_samples_leaves = []
        for tree in self.estimators_:
            self.train_samples_leaves.append(tree.apply(self.trained_X)) 

    def get_weights(self,x):   
        n_tree = len(self.estimators_)
        w=np.zeros((len(self.trained_X),))
        for t,tree in enumerate(self.estimators_):
            #train_samples_leaves = tree.apply(self.trained_X)
            train_samples_leaves = self.train_samples_leaves[t]
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
        y_pred = np.zeros((len(X),self.trained_y.shape[1]))
        for i in range(len(X)):
            x = X[i,:].reshape(1, -1)
            w = self.get_weights(x)
            selected_index = np.random.choice(a=list_index_train_X,size=1,replace=False,p=w)
            y_pred[i] = self.trained_y[selected_index]

        return np.array(y_pred)
