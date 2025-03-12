import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit,StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC,SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from drf import drf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb

from oversampling_strategies.categorical_oversampler import (
    MultiOutPutClassifier_and_MGS,
    NoSampling,
    WMGS_NC_cov,
)
from oversampling_strategies.continuous_oversampler import MGS2
from data.simulated_data import generate_initial_data_onecat_2025_02_28
from validation.classif_experiments import run_eval
#OracleOverSampler(random_state=0,generator=generate_initial_data_onecat,
#                            generator_params={'dimension':20,'random_state':24,'verbose':1})
from oversampling_strategies.forest_for_categorical import DrfSk, DrfSkExtraClassifier

##################################################
##################################################
categorical_handling = True
n_samples=5000
n_iter = 20

dimensions = [5,10,20,30,50,100,150,200]
for dimension in dimensions:
    K_MGS = dimension+1
    llambda_MGS = 1.0
    print('Value K_MGS : ', K_MGS)
    numeric_features =  np.arange(0,dimension,1)
    categorical_features = [-1]
    clf = lgb.LGBMClassifier(verbosity=-1,n_jobs=5,random_state=0)
    balanced_clf = lgb.LGBMClassifier(class_weight='balanced',verbosity=-1,n_jobs=5,random_state=0)
    #clf = LogisticRegression(penalty=None,random_state=0,max_iter=100)
    #balanced_clf = LogisticRegression(penalty=None,random_state=0,class_weight='balanced',max_iter=100)

    output_dir_path_subsampled = "../saved_experiments_categorial_features/2025-sim1/2025-02-28/no-scale/lgbm/K=d+1/5kSamples/ll1/startified/1fold/s4/normal/dimension_"+str(dimension) #drfsk-extra-max_f-1
    Path(output_dir_path_subsampled).mkdir(parents=True, exist_ok=True)
    init_name_file_subsampled = "2024-10-01-synthetic_"
    
    ##################################################
    ##################################################

    if categorical_handling :
        def to_str(x):
            return x.astype(str)


        def to_float(x):
            return x.astype(float)


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
        
    else:
        model=clf
        balanced_model = balanced_clf

    ################################
    print('******'*10)
    print('******'*10)
    print('Dimension : ',dimension)
    for i in range(n_iter):
        X_final, target_numeric, w_gauss = generate_initial_data_onecat_2025_02_28(dimension_continuous=dimension,n_samples=n_samples,random_state=i)
        X_final = X_final.astype(object)

        list_oversampling_and_params = [
            ("None", NoSampling(), {}, model),
            ("CW", NoSampling(), {}, balanced_model),
            ('ROS', RandomOverSampler(sampling_strategy="minority",random_state=i),{}, model),
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
                    Classifier=KNeighborsClassifier(n_neighbors=1),
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
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
                    Classifier=KNeighborsClassifier(n_neighbors=5),
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
                ),
                {},
                model,
            ),
            ('MGS-NC(mu)(d+1)(EmpCov)',WMGS_NC_cov(K=K_MGS,llambda=llambda_MGS,kind_cov = 'EmpCov',mucentered=True,version=1,categorical_features=categorical_features,random_state=i),{},model),
            ('MGS-NC(mu)(5)(EmpCov)',WMGS_NC_cov(K=5,llambda=llambda_MGS,kind_cov = 'EmpCov',mucentered=True,version=1,categorical_features=categorical_features,random_state=i),{},model),
            #(
            #    "MGS(mu)(d+1)(EmpCov) drf",
            #    MultiOutPutClassifier_and_MGS(
            #        K=K_MGS,
            #        llambda=llambda_MGS,
            #        categorical_features=categorical_features,
            #        Classifier=drf(min_node_size=1, num_trees=100, splitting_rule="CART",seed=i),
            #        random_state=i,
            #        kind_cov = 'EmpCov',
            #        mucentered=True,
            #        to_encode=False,
            #        to_encode_onehot=True,
            #        bool_rf=False,
            #        bool_rf_str=False,
            #        bool_rf_regressor=False,
            #        bool_drf=True,
            #        fit_nn_on_continuous_only= True,
            #    ),
            #    {},
            #    model,
            #),
            (
            "MGS(mu)(d+1)(EmpCov) DRFsk classique (mtry=None)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSk(random_state=i,n_jobs=5,max_features=None), #max_features=1.0
                random_state=i,
                kind_cov = 'EmpCov',
                mucentered=True,
                to_encode=False,
                to_encode_onehot=False,
                bool_rf=False,
                bool_rf_str=False,
                bool_rf_regressor=False,
                bool_drf=False,
                fit_nn_on_continuous_only= True,
            ),
            {},
            model,
            ),
            ]
        
        #splitter_stratified = ShuffleSplit(n_splits=5, test_size=.2, random_state=i)
        #splitter_stratified = ShuffleSplit(n_splits=1, test_size=.2, random_state=i)
        splitter_stratified = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=i)
        name_file = init_name_file_subsampled + str(i) + ".npy"
        run_eval(
            output_dir=output_dir_path_subsampled,
            name_file=name_file,
            X=X_final,
            y=target_numeric.ravel(),
            list_oversampling_and_params=list_oversampling_and_params,
            splitter=splitter_stratified,
            categorical_features=categorical_features,bool_to_save_data=True,bool_to_save_runing_time=True,
            y_splitter=None,#np.hstack((target_numeric.reshape(-1,1),w_gauss.reshape(-1,1))),
            to_standard_scale=False,
        )
print('END')








list_oversampling_and_params = [
            ("None", NoSampling(), {}, model),
            ("CW", NoSampling(), {}, balanced_model),
            ('ROS', RandomOverSampler(sampling_strategy="minority",random_state=i),{}, model),
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
                    Classifier=KNeighborsClassifier(n_neighbors=1),
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
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
                    Classifier=KNeighborsClassifier(n_neighbors=5),
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
                ),
                {},
                model,
            ),
            ('MGS-NC(mu)(d+1)(EmpCov)',WMGS_NC_cov(K=K_MGS,llambda=llambda_MGS,kind_cov = 'EmpCov',mucentered=True,version=1,categorical_features=categorical_features,random_state=i),{},model),
            ('MGS-NC(mu)(5)(EmpCov)',WMGS_NC_cov(K=5,llambda=llambda_MGS,kind_cov = 'EmpCov',mucentered=True,version=1,categorical_features=categorical_features,random_state=i),{},model),
            (
                "MGS(mu)(d+1)(EmpCov) drf",
                MultiOutPutClassifier_and_MGS(
                    K=K_MGS,
                    llambda=llambda_MGS,
                    categorical_features=categorical_features,
                    Classifier=drf(min_node_size=1, num_trees=100, splitting_rule="CART",seed=i),
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
                    to_encode=False,
                    to_encode_onehot=True,
                    bool_rf=False,
                    bool_rf_str=False,
                    bool_rf_regressor=False,
                    bool_drf=True,
                    fit_nn_on_continuous_only= True,
                ),
                {},
                model,
            ),
            ("None", NoSampling(), {}, model),
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
                    Classifier=KNeighborsClassifier(n_neighbors=1),
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
                ),
                {},
                model,
            ),
            (
            "MGS(mu)(d+1)(EmpCov) DRFsk classique (mtry=None)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSk(random_state=i,n_jobs=5,max_features=None), #max_features=1.0
                random_state=i,
                kind_cov = 'EmpCov',
                mucentered=True,
                to_encode=False,
                to_encode_onehot=False,
                bool_rf=False,
                bool_rf_str=False,
                bool_rf_regressor=False,
                bool_drf=False,
                fit_nn_on_continuous_only= True,
            ),
            {},
            model,
            ),
                (
                "MGS(mu)(d+1)(EmpCov) DRFskExtra classique (mtry=None)",
                MultiOutPutClassifier_and_MGS(
                    K=K_MGS,
                    llambda=llambda_MGS,
                    categorical_features=categorical_features,
                    Classifier=DrfSkExtraClassifier(random_state=i,n_jobs=5, n_estimators=100,max_features=None), #max_features=1.0
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
                    to_encode=False,
                    to_encode_onehot=False,
                    bool_rf=False,
                    bool_rf_str=False,
                    bool_rf_regressor=False,
                    bool_drf=False,
                    fit_nn_on_continuous_only= True,
                ),
                {},
                model,
            ),
        ]

