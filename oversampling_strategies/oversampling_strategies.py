import random
import math

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
            max(int(0.5*n_positifs),1),
            max(int(0.7*n_positifs),1)
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
        if n_points is None :
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
            mu = (1 / self.K+1) * X_positifs[indice_neighbors, :].sum(axis=0)
            sigma = (
                self.llambda
                * (1 / self.K+1)
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
        self, K, llambda, sampling_strategy="auto", random_state=None
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
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
        neighbors_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=False
        )

        n_synthetic_sample = n_final_sample - n_minoritaire


        # computing mu and covariance at once for every minority class points
        all_neighbors = X_positifs[neighbors_by_index.flatten()]
        mus = (1 / (self.K+1)) * all_neighbors.reshape(len(X_positifs), self.K + 1, dimension).sum(axis=1)
        centered_X = X_positifs[neighbors_by_index.flatten()] - np.repeat(mus, self.K + 1, axis=0)
        centered_X = centered_X.reshape(len(X_positifs), self.K + 1, dimension)
        covs = self.llambda * np.matmul(np.swapaxes(centered_X,1,2), centered_X) / (self.K+1)
        
        # spectral decomposition of all covariances
        eigen_values, eigen_vectors = np.linalg.eigh(covs) ## long
        eigen_values[eigen_values > 1e-10] = eigen_values[eigen_values > 1e-10] ** .5
        As = [eigen_vectors[i].dot(eigen_values[i]) for i in range(len(eigen_values))]

        np.random.seed(self.random_state)
        # sampling all new points
        #u = np.random.normal(loc=0, scale=1, size=(len(indices), dimension))
        #new_samples = [mus[central_point] + As[central_point].dot(u[central_point]) for i in indices]
        indices = np.random.randint(n_minoritaire, size=n_synthetic_sample)
        new_samples = np.zeros((n_synthetic_sample, dimension))
        for i,central_point in enumerate(indices):
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
class MGS_NC(BaseOverSampler):
    """
    MGS NC strategy 
    """

    def __init__(
        self, K, categorical_features,version,n_points=None, llambda=1.0,sampling_strategy="auto", random_state=None
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        if n_points is None :
            self.n_points = K
        else:
            self.n_points = n_points
        self.categorical_features = categorical_features
        self.version=version
        self.random_state = random_state 
        
    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        #X = _check_X(X)
        #self._check_n_features(X, reset=True)
        #self._check_feature_names(X, reset=True)
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
        bool_mask[self.categorical_features]= False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:,bool_mask] ## continuous features
        X_negatifs = X_negatifs_all_features[:,bool_mask] ## continuous features
        X_positifs_categorical = X_positifs_all_features[:,~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:,~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension_continuous = X_positifs.shape[1] ## of continuous features
        
        enc = OneHotEncoder(handle_unknown='ignore') ## encoding
        X_positifs_all_features_enc = enc.fit_transform(X_positifs_all_features).toarray()
        cste_med = np.median(np.sqrt(np.var(X_positifs,axis=0))) ## med constante from continuous variables
        if not math.isclose(cste_med,0):
            X_positifs_all_features_enc[:,dimension_continuous:] = cste_med / np.sqrt(2) # With one-hot encoding, the median will be repeated twice. We need
        # to divide by sqrt(2) such that we only have one median value
        # contributing to the Euclidean distance
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs_all_features_enc)
        neighbor_by_dist, neighbor_by_index = neigh.kneighbors(
            X=X_positifs_all_features_enc, n_neighbors=self.K + 1, return_distance=True
        )
        
        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, dimension_continuous))
        new_samples_cat = np.zeros((n_synthetic_sample, len(self.categorical_features)),dtype=object)

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
            mu = (1 / self.K +1) * X_positifs[indice_neighbors, :].sum(axis=0)
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
            if self.version==1: ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    most_common = Counter(X_positifs_categorical[indice_neighbors,cat_feature]).most_common(1)[0][0]
                    new_samples_cat[i, cat_feature] = most_common
            elif self.version==2: ## sampling of one of the nearest neighbors per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(X_positifs_categorical[indice_neighbors,cat_feature],replace=False) 
            elif self.version==3: ## sampling of one of the nearest neighbors per categorical feature using dsitance
                #### We take the nn of the central point. The latter is excluded
                epsilon_weigths_sampling = 10e-6
                indice_neighbors_without_0 = np.arange(start=1,stop=self.K + 1,dtype=int)
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(X_positifs_categorical[indice_neighbors_without_0,cat_feature],replace=False,
                                                                      p=((1/(neighbor_by_dist[indice][indice_neighbors_without_0]+epsilon_weigths_sampling)) / (1/(neighbor_by_dist[indice][indice_neighbors_without_0]+epsilon_weigths_sampling)).sum() ))
            else :
                raise ValueError(
                    "Selected version not allowed "
                    "Please chose an existing version"
                )
        np.random.seed()
        
        ##### END ######
        new_samples_final = np.zeros((n_synthetic_sample,X_positifs_all_features.shape[1]),dtype=object)
        new_samples_final[:,bool_mask] = new_samples
        new_samples_final[:,~bool_mask] = new_samples_cat
        
        X_positifs_final = np.zeros((len(X_positifs),X_positifs_all_features.shape[1]),dtype=object)
        X_positifs_final[:,bool_mask] = X_positifs
        X_positifs_final[:,~bool_mask] = X_positifs_categorical
        
        X_negatifs_final = np.zeros((len(X_negatifs),X_positifs_all_features.shape[1]),dtype=object)
        X_negatifs_final[:,bool_mask] = X_negatifs
        X_negatifs_final[:,~bool_mask] = X_negatifs_categorical
        
        oversampled_X = np.concatenate((X_negatifs_final, X_positifs_final, new_samples_final), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y



class MGS_cat(BaseOverSampler):
    """
    MGS NC distance-without-discrete-features 
    """

    def __init__(
        self, K,categorical_features,version,n_points=None, llambda=10,sampling_strategy="auto", random_state=None
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        if n_points is None :
            self.n_points = K
        else:
            self.n_points = n_points
        self.categorical_features = categorical_features
        self.version=version
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
        bool_mask[self.categorical_features]= False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:,bool_mask] ## continuous features
        X_negatifs = X_negatifs_all_features[:,bool_mask] ## continuous features
        X_positifs_categorical = X_positifs_all_features[:,~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:,~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension = X_positifs.shape[1] ## features continues seulement
        
        ######### CONTINUOUS ################
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbor_by_dist,neighbor_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=True
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, dimension))
        new_samples_cat = np.zeros((n_synthetic_sample, len(self.categorical_features)),dtype=object)
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
            mu = (1 / self.K+1) * X_positifs[indice_neighbors, :].sum(axis=0)
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
            if self.version==1: ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):                    
                    most_common = Counter(X_positifs_categorical[indice_neighbors,cat_feature]).most_common(1)[0][0]
                    new_samples_cat[i, cat_feature] = most_common
            elif self.version==2: ## sampling of one of the nearest neighbors per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(X_positifs_categorical[indice_neighbors,cat_feature],replace=False)
            elif self.version==3: ## sampling of one of the nearest neighbors per categorical feature using dsitance
                #### We take the nn of the central point. The latter is excluded
                epsilon_weigths_sampling = 10e-6
                indice_neighbors_without_0 = np.arange(start=1,stop=self.K + 1,dtype=int)
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(X_positifs_categorical[indice_neighbors_without_0,cat_feature],replace=False,
                                                                      p=((1/(neighbor_by_dist[indice][indice_neighbors_without_0]+epsilon_weigths_sampling)) / (1/(neighbor_by_dist[indice][indice_neighbors_without_0]+epsilon_weigths_sampling)).sum() ))
            else :
                raise ValueError(
                    "Selected version not allowed "
                    "Please chose an existing version"
                )
        np.random.seed() 
        
        ##### END ######
        new_samples_final = np.zeros((n_synthetic_sample,X_positifs_all_features.shape[1]),dtype=object)
        new_samples_final[:,bool_mask] = new_samples
        new_samples_final[:,~bool_mask] = new_samples_cat
        
        X_positifs_final = np.zeros((len(X_positifs),X_positifs_all_features.shape[1]),dtype=object)
        X_positifs_final[:,bool_mask] = X_positifs
        X_positifs_final[:,~bool_mask] = X_positifs_categorical
        
        X_negatifs_final = np.zeros((len(X_negatifs),X_positifs_all_features.shape[1]),dtype=object)
        X_negatifs_final[:,bool_mask] = X_negatifs
        X_negatifs_final[:,~bool_mask] = X_negatifs_categorical
        
        oversampled_X = np.concatenate((X_negatifs_final, X_positifs_final, new_samples_final), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y

class SMOTE_cat(BaseOverSampler):
    """
    SMOTE NC  distance-without-discrete-features 
    """

    def __init__(
        self, K,categorical_features,version,sampling_strategy="auto", random_state=None
    ):
        """
        K : int.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.categorical_features = categorical_features
        self.version=version
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
        bool_mask[self.categorical_features]= False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:,bool_mask] ## continuous features
        X_negatifs = X_negatifs_all_features[:,bool_mask] ## continuous features
        X_positifs_categorical = X_positifs_all_features[:,~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:,~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension = X_positifs.shape[1] ## features continues seulement
        
        ######### CONTINUOUS ################
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbor_by_dist, neighbor_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=True
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, dimension))
        new_samples_cat = np.zeros((n_synthetic_sample, len(self.categorical_features)),dtype=object)
        np.random.seed(self.random_state)
        for i in range(n_synthetic_sample):
            indice = np.random.randint(n_minoritaire) #individu centrale qui sera choisi
            indice_neigh = np.random.randint(1,self.K+1)# Sélection voisin parmi les K. On exclu 0 car c indice lui-même
            indice_neighbor = neighbor_by_index[indice][indice_neigh]
            w = np.random.uniform(0,1) #facteur alpha de la première difference
            new_samples[i,:] = X_positifs[indice] + w * (X_positifs[indice_neighbor] - X_positifs[indice])
        ############### CATEGORICAL ##################
            indice_neighbors_with_0 = np.arange(start=0,stop=self.K + 1,dtype=int)
            if self.version==1:  ## the most common occurence is chosen per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    most_common = Counter(X_positifs_categorical[indice_neighbors_with_0,cat_feature]).most_common(1)[0][0]
                    new_samples_cat[i, cat_feature] = most_common
            elif self.version==2: ## sampling of one of the nearest neighbors per categorical feature
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(X_positifs_categorical[indice_neighbors_with_0,cat_feature],replace=False)
            elif self.version==3: ## sampling of one of the nearest neighbors per categorical feature using dsitance
                #### We take the nn of the central point. The latter is excluded
                epsilon_weigths_sampling = 10e-6
                indice_neighbors_without_0 = np.arange(start=1,stop=self.K + 1,dtype=int)
                for cat_feature in range(len(self.categorical_features)):
                    new_samples_cat[i, cat_feature] = np.random.choice(X_positifs_categorical[indice_neighbors_without_0,cat_feature],replace=False,
                                                                      p=( (1/(neighbor_by_dist[indice][indice_neighbors_without_0]+epsilon_weigths_sampling)) / (1/(neighbor_by_dist[indice][indice_neighbors_without_0]+epsilon_weigths_sampling)).sum() ))
            else :
                raise ValueError(
                    "Selected version not allowed "
                    "Please chose an existing version"
                )
                
        np.random.seed() 
        
        ##### END ######
        new_samples_final = np.zeros((n_synthetic_sample,X_positifs_all_features.shape[1]),dtype=object)
        new_samples_final[:,bool_mask] = new_samples
        new_samples_final[:,~bool_mask] = new_samples_cat

        X_positifs_final = np.zeros((len(X_positifs),X_positifs_all_features.shape[1]),dtype=object)
        X_positifs_final[:,bool_mask] = X_positifs
        X_positifs_final[:,~bool_mask] = X_positifs_categorical
        
        X_negatifs_final = np.zeros((len(X_negatifs),X_positifs_all_features.shape[1]),dtype=object)
        X_negatifs_final[:,bool_mask] = X_negatifs
        X_negatifs_final[:,~bool_mask] = X_negatifs_categorical
        
        oversampled_X = np.concatenate((X_negatifs_final, X_positifs_final, new_samples_final), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y

    
class MultiOutPutClassifier_and_MGS(BaseOverSampler):
    """
    MultiOutPutClassifier and MGS
    """

    def __init__(
        self, K,categorical_features,Classifier,to_encode=False,n_points=None, llambda=1.0,sampling_strategy="auto", random_state=None
    ):
        """
        llambda is a float.
        """
        super().__init__(sampling_strategy=sampling_strategy)
        self.K = K
        self.llambda = llambda
        if n_points is None :
            self.n_points = K
        else:
            self.n_points = n_points
        self.categorical_features = categorical_features
        self.Classifier = Classifier
        self.random_state = random_state
        self.to_encode=to_encode
        
    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        #X = _check_X(X)
        #self._check_n_features(X, reset=True)
        #self._check_feature_names(X, reset=True)
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
        if len(self.categorical_features) == X.shape[1]:
            raise ValueError(
                "MultiOutPutClassifier_and_MGS is not designed to work only with categorical "
                "features. It requires some numerical features."
            )
                
        bool_mask = np.ones((X_positifs.shape[1]), dtype=bool)
        bool_mask[self.categorical_features]= False
        X_positifs_all_features = X_positifs.copy()
        X_negatifs_all_features = X_negatifs.copy()
        X_positifs = X_positifs_all_features[:,bool_mask] ## continuous features
        X_negatifs = X_negatifs_all_features[:,bool_mask] ## continuous features
        X_positifs_categorical = X_positifs_all_features[:,~bool_mask]
        X_negatifs_categorical = X_negatifs_all_features[:,~bool_mask]
        X_positifs = X_positifs.astype(float)

        n_minoritaire = X_positifs.shape[0]
        dimension = X_positifs.shape[1] ## features continues seulement

        np.random.seed(self.random_state)
        if self.to_encode:
            ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,dtype=float)
            X_positifs_categorical_encoded = ord_encoder.fit_transform(X_positifs_categorical.astype(str)) 
            ### Fit :
            self.Classifier.fit(X_positifs,X_positifs_categorical_encoded) # learn on continuous features in order to predict categorical feature  
        else:
            self.Classifier.fit(X_positifs,X_positifs_categorical.astype(str)) # learn on continuous features in order to predict categorical features
        ######### CONTINUOUS ################
        neigh = NearestNeighbors(n_neighbors=self.K, algorithm="ball_tree")
        neigh.fit(X_positifs)
        neighbor_by_index = neigh.kneighbors(
            X=X_positifs, n_neighbors=self.K + 1, return_distance=False
        )

        n_synthetic_sample = n_final_sample - n_minoritaire
        new_samples = np.zeros((n_synthetic_sample, dimension))
        #new_samples_cat = np.zeros((n_synthetic_sample, len(self.categorical_features)),dtype=object)
        for i in range(n_synthetic_sample):
            indice = np.random.randint(n_minoritaire)
            indices_neigh = [
                0
            ]  ## the central point is selected for the expectation and covariance matrix
            indices_neigh.extend(
                random.sample(range(1, self.K + 1), self.n_points)
            )  # The nearrest neighbor selected for the estimation
            indice_neighbors = neighbor_by_index[indice][indices_neigh]
            mu = (1 / self.K+1) * X_positifs[indice_neighbors, :].sum(axis=0)
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
            
        #    if self.to_encode:
        #        new_pred= self.Classifier.predict(newdata = new_observation,functional = "mean")
        #        new_pred = new_pred.mean.ravel()
        #    else:
        #        new_pred = self.Classifier.predict(new_observation.reshape(1, -1))
        #    new_samples_cat[i, :] = new_pred
        if self.to_encode:
            out = self.Classifier.predict(newdata = new_samples, functional = "mean")
            new_samples_cat = out.mean
        else:
             new_samples_cat = self.Classifier.predict(new_samples)
        np.random.seed()
        ##### END ######
        if self.to_encode:
            new_samples_cat = ord_encoder.inverse_transform(new_samples_cat.astype(int))
        new_samples_final = np.zeros((n_synthetic_sample,X_positifs_all_features.shape[1]),dtype=object)
        new_samples_final[:,bool_mask] = new_samples
        new_samples_final[:,~bool_mask] = new_samples_cat
        #new_samples_final = np.concatenate((new_samples,new_samples_cat), axis=1)
        
        X_positifs_final = np.zeros((len(X_positifs),X_positifs_all_features.shape[1]),dtype=object)
        X_positifs_final[:,bool_mask] = X_positifs
        X_positifs_final[:,~bool_mask] = X_positifs_categorical
        
        X_negatifs_final = np.zeros((len(X_negatifs),X_positifs_all_features.shape[1]),dtype=object)
        X_negatifs_final[:,bool_mask] = X_negatifs
        X_negatifs_final[:,~bool_mask] = X_negatifs_categorical
        
        oversampled_X = np.concatenate((X_negatifs_final, X_positifs_final, new_samples_final), axis=0)
        oversampled_y = np.hstack(
            (np.full(len(X_negatifs), 0), np.full((n_final_sample,), 1))
        )

        return oversampled_X, oversampled_y



# SMOTE ENC from authors :
from sklearn.utils import check_array, _safe_indexing, sparsefuncs_fast, check_random_state
#from scipy import stats
from numbers import Integral
from scipy import sparse
import pandas as pd
from sklearn import clone
from sklearn.neighbors._base import KNeighborsMixin
from imblearn.exceptions import raise_isinstance_error
class SMOTE_ENC():
    
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        
    def chk_neighbors(self, nn_object, additional_neighbor):
        if isinstance(nn_object, Integral):
            return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
        elif isinstance(nn_object, KNeighborsMixin):
            return clone(nn_object)
        else:
            raise_isinstance_error(nn_name, [int, KNeighborsMixin], nn_object)  ### A regerder en détail     
    
    def generate_samples(self, X, nn_data, nn_num, rows, cols, steps, continuous_features_,):
        rng = check_random_state(42)

        diffs = nn_data[nn_num[rows, cols]] - X[rows]

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs 

        X_new = (X_new.tolil() if sparse.issparse(X_new) else X_new)
        # convert to dense array since scipy.sparse doesn't handle 3D
        nn_data = (nn_data.toarray() if sparse.issparse(nn_data) else nn_data)

        all_neighbors = nn_data[nn_num[rows]]

        for idx in range(continuous_features_.size, X.shape[1]):

            mode = stats.mode(all_neighbors[:, :, idx], axis = 1)[0]

            X_new[:, idx] = np.ravel(mode)            
        return X_new
    
    def make_samples(self, X, y_dtype, y_type, nn_data, nn_num, n_samples, continuous_features_, step_size=1.0):
        random_state = check_random_state(42)
        samples_indices = random_state.randint(low=0, high=len(nn_num.flatten()), size=n_samples)    
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self.generate_samples(X, nn_data, nn_num, rows, cols, steps, continuous_features_)
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
        imb_ratio = len(X_filtered)/len(X)
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
            nan_idx_array = np.ravel(np.argwhere(np.isnan(X.loc[:, column]).to_numpy())) ## Add .tonumpy() Abd
            if len(nan_idx_array) > 0 :
                nan_dict[c] = nan_idx_array
            c = c + 1
            X.loc[:, column].fillna(-1, inplace = True)
            
        X.drop(target_column, axis=1, inplace=True)
        return X, encoded_dict_list, nan_dict

    def fit_resample(self, X, y):
        X = pd.DataFrame(X) ## ABD
        y = pd.DataFrame({"target":y}) ##ABD
        X_cat_encoded, encoded_dict_list, nan_dict = self.cat_corr_pandas(X.iloc[:,np.asarray(self.categorical_features)], y, target_column='target', target_value=1)
#         X_cat_encoded = np.ravel(np.array(X_cat_encoded))
        X_cat_encoded = np.array(X_cat_encoded)
        y = np.ravel(y)
        X = np.array(X)

        unique, counts = np.unique(y, return_counts=True)
        target_stats = dict(zip(unique, counts))
        n_sample_majority = max(target_stats.values())
        class_majority = max(target_stats, key=target_stats.get)
        sampling_strategy = {key: n_sample_majority - value for (key, value) in target_stats.items() if key != class_majority}

        n_features_ = X.shape[1]
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == 'bool':
            categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any([cat not in np.arange(n_features_) for cat in categorical_features]):
                raise ValueError('Some of the categorical indices are out of range. Indices'
                            ' should be between 0 and {}'.format(n_features_))
            categorical_features_ = categorical_features

        continuous_features_ = np.setdiff1d(np.arange(n_features_),categorical_features_)

        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=['csr', 'csc'])
        X_minority = _safe_indexing(X_continuous, np.flatnonzero(y == class_minority))

        if sparse.issparse(X):
            if X.format == 'csr':
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
        X_resampled = X_encoded.copy()
        y_resampled = y.copy()


        for class_sample, n_samples in sampling_strategy.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X_encoded, target_class_indices)
            nn_k_ = self.chk_neighbors(5, 1)
            nn_k_.fit(X_class)

            nns = nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self.make_samples(X_class, y.dtype, class_sample, X_class, nns, n_samples, continuous_features_, 1.0)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
                sparse_func = 'tocsc' if X.format == 'csc' else 'tocsr'
                X_resampled = getattr(X_resampled, sparse_func)()
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))
            
        X_resampled_copy = X_resampled.copy()
        i = 0
        for col in range(continuous_features_.size, X.shape[1]):
            encoded_dict = encoded_dict_list[i]
            i = i + 1
            for key, value in encoded_dict.items():
                X_resampled_copy[:, col] = np.where(np.round(X_resampled_copy[:, col], 4) == np.round(value * median_std_, 4), key, X_resampled_copy[:, col])

        for key, value in nan_dict.items():
            for item in value:
                X_resampled_copy[item, continuous_features_.size + key] = X_copy[item, continuous_features_.size + key]

               
        X_resampled = X_resampled_copy   
        indices_reordered = np.argsort(np.hstack((continuous_features_, categorical_features_)))
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
        ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,dtype=int)
        X[:,self.categorical_features] = ord_encoder.fit_transform(X[:,self.categorical_features]) 
        ### Sampling :
        X_res, y_res = super().fit_resample(X,y)
        X_res[:,self.categorical_features] = ord_encoder.inverse_transform(X_res[:,self.categorical_features].astype(int)) 
        return X_res, y_res