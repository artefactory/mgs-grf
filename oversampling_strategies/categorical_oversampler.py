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
from collections import namedtuple ## KNN

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


ModeResult = namedtuple('ModeResult', ('mode', 'count'))
def mode_rand(a, axis):
    in_dims = list(range(a.ndim))
    a_view = np.transpose(a, in_dims[:axis] + in_dims[axis+1:] + [axis])

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
            X_positifs_all_features_enc[:, dimension_continuous:] =  X_positifs_all_features_enc[:, dimension_continuous:] * (cste_med / np.sqrt(2))  
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
            indices_neigh = np.arange(1,self.K+1,1)
            indice_neighbors = neighbor_by_index[central_point][indices_neigh]

            if (
                self.version == 1
            ):  ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    most_common,_ = mode_rand(X_positifs_categorical[indice_neighbors, cat_feature].ravel(),axis=0)
                    new_samples_cat[i, cat_feature] = most_common[0]
                    #vals, cnts = np.unique(X_positifs_categorical[indice_neighbors, cat_feature].ravel(), return_counts=True)
                    #ind_maxes = np.flatnonzero(cnts == np.max(cnts))
                    #selected_ind = np.random.choice(ind_maxes)
                    #new_samples_cat[i, cat_feature] = vals[selected_ind]
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
        bool_rf_str=False,
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
        self.bool_rf_str = bool_rf_str ## Do not use with encoding
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
        
    def array_of_lists_to_array(self,arr): ## Used when calling fit_resampl with  bool_rf_str=True
        return np.apply_along_axis(lambda a: np.array(a[0]), -1, arr[..., None])
    
    def fit_resample(self, X, y,to_return_classifier=False,scaler=None):  # scaler Necessary only for SemiOracle
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
            output = self._fit_resample(X, y,to_return_classifier=to_return_classifier)
        else:
            output = self._fit_resample(X, y,scaler=scaler)

        X_,y_=output[0],output[1]
        if len(output) == 2:
            return (X_, y_) 
        elif len(output) == 3:
            return (X_, y_, output[2])
        else:
            return (X_, y_, output[2],output[3])
        #return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    def _fit_resample(self, X, y=None,to_return_classifier=False,scaler=None, n_final_sample=None):
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

        elif self.bool_rf_str : #case RFc on the concatenated strings modalities
            #type_cat = X_positifs_categorical.astype(str).dtype
            sep_array = np.full((n_minoritaire,len(self.categorical_features)-1),',',dtype=str)
            sep_array = np.hstack((sep_array,np.full((n_minoritaire,1),'',dtype=str))) # We do not want an comma after the last modality
            X_positifs_categorical_str= np.char.add(X_positifs_categorical.astype(str),sep_array) # We add commas at the end of each mdality
            X_positifs_categorical_str = X_positifs_categorical_str.astype(object).sum(axis=1) # We concatenate by row the modalities 
            self.Classifier.fit(
                X_positifs, X_positifs_categorical_str
            )  # learn on continuous features in order to predict categorical features combinasion concatenated

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
                X_positifs_all_features_enc[:, dimension_continuous:] =  X_positifs_all_features_enc[:, dimension_continuous:] * (cste_med / np.sqrt(2)) 
                  # With one-hot encoding, the median will be repeated twice. We need
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

        elif self.bool_rf and self.to_encode_onehot:
            new_samples_cat = self.Classifier.predict(new_samples)

        elif self.bool_rf_str:
            if self.to_encode_onehot:
                raise ValueError(
                        "Cannot apply bool_rf_str=True with to_encode_onehot=True"
                    )
            new_samples_cat = self.Classifier.predict(new_samples)
            new_samples_cat = np.char.split(new_samples_cat.astype(str),sep=',') ## We split the predictions
            new_samples_cat = self.array_of_lists_to_array(new_samples_cat) ## We get back the right shape for categorical features array

        elif self.bool_drfsk_regressor and self.to_encode_onehot:
            new_samples_cat_pred= self.Classifier.predict(new_samples) # list of  pred for each categorical one hot encoded.
            new_samples_cat = var_scaler_cat.inverse_transform(new_samples_cat_pred)
        elif self.bool_rf_regressor and self.to_encode_onehot:
            categories_onehot = onehot_encoder.categories_
            new_samples_cat_pred= self.Classifier.predict(new_samples) # list of  pred for each categorical one hot encoded.
            array_pred_by_tree = np.zeros((len(self.Classifier.estimators_),new_samples_cat_pred.shape[0],new_samples_cat_pred.shape[1]))
            for t,tree in enumerate(self.Classifier.estimators_):
                curr_tree_pred = tree.predict(new_samples)
                curr_tree_pred = var_scaler_cat.inverse_transform(curr_tree_pred)
                array_pred_by_tree[t,:,:] = curr_tree_pred
            new_samples_cat_pred_probas = np.mean(array_pred_by_tree,axis=0)

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
            new_samples_cat = ord_encoder.inverse_transform(new_samples_cat)
        elif self.to_encode_onehot:
            new_samples_cat = onehot_encoder.inverse_transform(new_samples_cat)
            
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

        if to_return_classifier:
            if self.to_encode_onehot:
                return oversampled_X, oversampled_y, self.Classifier, onehot_encoder
            else:
                return oversampled_X, oversampled_y,self.Classifier
        else:
            return oversampled_X, oversampled_y


from imblearn.over_sampling import SMOTENC
from imblearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils import (check_random_state)
from scipy import sparse

class SMOTENC2(SMOTENC):
    _required_parameters = ["categorical_features"]

    _parameter_constraints: dict = {
        **SMOTE._parameter_constraints,
        "categorical_features": ["array-like", StrOptions({"auto"})],
        "categorical_encoder": [
            HasMethods(["fit_transform", "inverse_transform"]),
            None,
        ],
    }

    def __init__(
        self,
        categorical_features,
        *,
        categorical_encoder=None,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super(SMOTE,self).__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.categorical_features = categorical_features
        self.categorical_encoder = categorical_encoder
    
    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps, y_type, y=None):
        """Generate a synthetic sample with an additional steps for the
        categorical features.

        Each new sample is generated the same way than in SMOTE. However, the
        categorical features are mapped to the most frequent nearest neighbors
        of the majority class.
        """
        rng = check_random_state(self.random_state)
        X_new = super(SMOTE,self)._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        n_new_samples =X_new.shape[0]
        # change in sparsity structure more efficient with LIL than CSR
        X_new = X_new.tolil() if sparse.issparse(X_new) else X_new

        # convert to dense array since scipy.sparse doesn't handle 3D
        nn_data = nn_data.toarray() if sparse.issparse(nn_data) else nn_data

        # In the case that the median std was equal to zeros, we have to
        # create non-null entry based on the encoded of OHE
        if math.isclose(self.median_std_[y_type], 0):
            nn_data[
                :, self.continuous_features_.size :
            ] = self._X_categorical_minority_encoded

        all_neighbors = nn_data[nn_num[rows]]

        categories_size = [self.continuous_features_.size] + [
            cat.size for cat in self.categorical_encoder_.categories_
        ]

        for start_idx, end_idx in zip(
            np.cumsum(categories_size)[:-1], np.cumsum(categories_size)[1:]
        ):
            col_maxs = all_neighbors[:, :, start_idx:end_idx].sum(axis=1)
            # tie breaking argmax
            #is_max = np.isclose(col_maxs, col_maxs.max(axis=1, keepdims=True))
            #max_idxs = rng.permutation(np.argwhere(is_max))
            #max_idxs = np.random.permutation(np.argwhere(is_max))
            #xs, idx_sels = np.unique(max_idxs[:, 0], return_index=True)
            #col_sels = max_idxs[idx_sels, 1]
            col_sels = []
            for i in range(n_new_samples):
                sum_neigh =np.sum(all_neighbors[i, :, start_idx:end_idx],axis=0).ravel()
                np.isclose(sum_neigh, sum_neigh.max())
                ind_maxes = np.flatnonzero(sum_neigh == np.max(sum_neigh))
                selected_col = np.random.choice(ind_maxes)
                col_sels.append(selected_col)
                #most_common,_ = mode_rand(all_neighbors[i, :, start_idx:end_idx],axis=0)
                #print('all_neighbors : ', all_neighbors[i, :, start_idx:end_idx])
                #print('most_common : ',most_common)
                #selected_col=np.argmax(most_common[0]).ravel()[0]
                #print('selected_col : ',selected_col)
                #col_sels.append(selected_col)
                #list_ind_col = np.argwhere(is_max[i,:]).ravel()
                #n_mod = len(list_ind_col)

                #if n_mod == 1:
                #    col_sels.append(list_ind_col[0])
                #else:
                #    selected_col = np.random.choice(list_ind_col)
                #    col_sels.append(selected_col)

            col_sels = np.array(col_sels)
            xs = np.arange(start=0,stop=n_new_samples,step=1)
            ys = start_idx + col_sels
            X_new[:, start_idx:end_idx] = 0
            X_new[xs, ys] = 1
            
        return X_new