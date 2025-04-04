import random
import math

import numpy as np

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_target_type

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
    

from data.data import generate_synthetic_features_multinomial_quadruple
class OracleTwoCat5():
    def __init__(self):
        pass
    def fit(self,X,y):
        pass
    def predict(self,X,scaler,y=None):
        inversed_X = scaler.inverse_transform(X)
        n_samples = len(inversed_X)
        
        z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial_quadruple(
        X=inversed_X,index_informatives=[0,1,2],list_modalities=['Ae','Bd','Af','Ce'],beta1=np.array([1,3,2]),
        beta2=np.array([4,-7,3]),beta3=np.array([5,-1,6]),beta4=np.array([3,2,1]),intercept=-3
        )
        first_cat_feature = np.array([z_feature_cat_uniform_numeric[i][0] for i in range(n_samples) ])
        second_cat_feature = np.array([z_feature_cat_uniform_numeric[i][1] for i in range(n_samples) ])
        feature_cat_uniform_numeric_final = np.hstack((first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
                                                                                                
        return feature_cat_uniform_numeric_final
    
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