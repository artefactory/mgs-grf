import os
import sys


sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
# from drf import drf

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from oversampling_strategies.categorical_oversampler import (
    NoSampling,
    MultiOutPutClassifier_and_MGS,
    WMGS_NC_cov,
)
from oversampling_strategies.forest_for_categorical import DrfSk, KNNTies
# from sklearn.neighbors import Ne


from validation.classif_experiments import run_eval, read_subsampling_indices, subsample_to_ratio_indices
from data.data import load_BankChurners_data


def to_str(x):
    return x.astype(str)


def to_float(x):
    return x.astype(float)


################# INitialisation #################

#initial_X, initial_y = load_BankMarketing_data()
#numeric_features = [0, 5, 11, 12, 13, 14]
#categorical_features = [1, 2, 3, 4, 6, 7, 8, 9, 10, 15]

initial_X,initial_y = load_BankChurners_data()
numeric_features = [0,2,7,8,9,10,11,12,13,14,15,16,17,18]
categorical_features = [1,3,4,5,6]

##################################
K_MGS = max(len(numeric_features) + 1, 5)
llambda_MGS = 1.0
print("Value K_MGS : ", K_MGS)
print("llambda_MGS  : ", llambda_MGS)

clf = lgb.LGBMClassifier(n_estimators=100,verbosity=-1, n_jobs=8, random_state=0)
balanced_clf = lgb.LGBMClassifier(
    n_estimators=100,class_weight="balanced", verbosity=-1, n_jobs=8, random_state=0
)
n_iter = 20

# output_dir_path =  "../saved_experiments_categorial_features/2024-06-19_BankMarketing"
# indices_kept_1 = subsample_to_ratio_indices(X=X,y=y,ratio=0.01,seed_sub=5,
#    output_dir_subsampling=output_dir_path,
#    name_subsampling_file='bankmarketing_sub_original_to_1')


output_dir_path = "../saved_experiments_categorial_features/2024-06-19_BankChurners"
indices_kept_1 = subsample_to_ratio_indices(X=initial_X,y=initial_y,ratio=0.01,seed_sub=5,
                                            output_dir_subsampling=output_dir_path,
                                            name_subsampling_file='BankChurners_sub_original_to_1')

if True:
    X, y = read_subsampling_indices(
        X=initial_X,
        y=initial_y,
        dir_subsampling="../saved_experiments_categorial_features/2024-06-19_BankChurners",
        name_subsampling_file="BankChurners_sub_original_to_1",
        get_indexes=False,
    )
else:
    X, y = initial_X, initial_y

output_dir_path = "../saved_experiments_categorial_features/2024-06-19_BankChurners/2025/subsample_to_1"
Path(output_dir_path).mkdir(parents=True, exist_ok=True)
init_name_file_original = "2024-11-30-lgbm_"


###############################################################
########################### RUN ###############################
###############################################################
fun_tr_str = FunctionTransformer(to_str)
fun_tr_float = FunctionTransformer(to_float)
numeric_transformer = Pipeline(steps=[("Transform_float", fun_tr_float)])
categorical_transformer = Pipeline(
    steps=[
        ("Transform_str", fun_tr_str),
        ("OneHot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = Pipeline(steps=[("preprocessor", preprocessor), ("rf", clf)])
balanced_model = Pipeline(steps=[("preprocessor", preprocessor), ("rf", balanced_clf)])

# Initial run
for i in range(n_iter):
    list_oversampling_and_params = [
        ("None", NoSampling(), {}, model),
        ("CW", NoSampling(), {}, balanced_model),
        (
            "ROS",
            RandomOverSampler(sampling_strategy="minority", random_state=i),
            {},
            model,
        ),
        (
            "RUS",
            RandomUnderSampler(
                sampling_strategy="majority", replacement=False, random_state=i
            ),
            {},
            model,
        ),
        (
            "SmoteNC (K=5)",
            SMOTENC(
                k_neighbors=5, categorical_features=categorical_features, random_state=i
            ),
            {},
            model,
        ),
        (
            "MGS(mu)(d+1)(EmpCov) 1-NN",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=KNNTies(n_neighbors=1),
                random_state=i,
                kind_cov="EmpCov",
                mucentered=True,
                fit_nn_on_continuous_only=True,
            ),
            {},
            model,
        ),
        (
            "MGS(mu)(d+1)(EmpCov) 5-NN",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=KNNTies(n_neighbors=5),
                random_state=i,
                kind_cov="EmpCov",
                mucentered=True,
                fit_nn_on_continuous_only=True,
            ),
            {},
            model,
        ),
        (
            "MGS-NC(mu)(d+1)(EmpCov)",
            WMGS_NC_cov(
                K=K_MGS,
                llambda=llambda_MGS,
                kind_cov="EmpCov",
                mucentered=True,
                version=1,
                categorical_features=categorical_features,
                random_state=i,
            ),
            {},
            model,
        ),
        (
            "MGS(mu)(d+1)(EmpCov) DRFsk classique (mtry=def=sqrt)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSk(random_state=i, n_jobs=5),
                random_state=i,
                kind_cov="EmpCov",
                mucentered=True,
                to_encode=False,
                to_encode_onehot=False,
                bool_rf=False,
                bool_rf_str=False,
                bool_rf_regressor=False,
                bool_drf=False,
                fit_nn_on_continuous_only=True,
            ),
            {},
            model,
        ),
    ]

    splitter_stratified = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=100 + i
    )
    name_file = init_name_file_original + str(i) + ".npy"
    run_eval(
        output_dir=output_dir_path,
        name_file=name_file,
        X=X,
        y=y,
        list_oversampling_and_params=list_oversampling_and_params,
        splitter=splitter_stratified,
        categorical_features=categorical_features,
        bool_to_save_data=True,
        bool_to_save_runing_time=True,
    )
    print("FIN Iteration :", i)
