import os
import sys


sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTEN
from drf import drf

from sklearn.ensemble import RandomForestClassifier 
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTENC 
from oversampling_strategies.categorical_oversampler import (
    NoSampling,
    MultiOutPutClassifier_and_MGS,
    WMGS_NC_cov,
)
from oversampling_strategies.forest_for_categorical import (DrfSk,DrfSkRegressor,DrfSkExtraClassifier,KNNTies)
#from sklearn.neighbors import Ne


from validation.classif_experiments import run_eval,read_subsampling_indices
from data.data import load_feedzai_data

def to_str(x):
  return x.astype(str)

def to_float(x):
  return x.astype(float)


################# INitialisation #################

initial_X,initial_y = load_feedzai_data()
categorical_features  = [4,7,14,16,17,18,19,20,21,23,24,26,27,28,29,30]
numeric_features= [0,1,2,3,5,6,8,9,10,11,12,13,15,22,25]

#initial_X,initial_y = load_BankMarketing_data()
#numeric_features = [0,5,11,12,13,14]
#categorical_features = [1,2,3,4,6,7,8,9,10,15]

#initial_X,initial_y = load_defaultUCI_data()
#numeric_features = [0,4,11,12,13,14,15,16,17,18,19,20,21,22]
#categorical_features = [1,2,3,5,6,7,8,9,10]

#initial_X,initial_y = load_BankChurners_data()
#numeric_features = [0,2,7,8,9,10,11,12,13,14,15,16,17,18]
#categorical_features = [1,3,4,5,6]

#initial_X,initial_y = load_TelcoChurn_data()
#numeric_features = [4,17,18]
#categorical_features = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16]

#initial_X,initial_y = load_census_data()
#categorical_features = [1,3,5,6,7,8,9,13]
#numeric_features = [0,2,4,10,11,12]

##################################
K_MGS = max(len(numeric_features)+1, 5)
llambda_MGS = 1.0
print('Value K_MGS : ', K_MGS)
print('llambda_MGS  : ',llambda_MGS) 


clf = lgb.LGBMClassifier(verbosity=-1,n_jobs=8,random_state=0)
balanced_clf = lgb.LGBMClassifier(class_weight='balanced',verbosity=-1,n_jobs=8,random_state=0)
#clf = RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs=5)
#balanced_clf = RandomForestClassifier(n_estimators=100,criterion='gini',class_weight='balanced',n_jobs=5)
n_iter = 20



if False:
    X, y = read_subsampling_indices(
    X=initial_X,
    y=initial_y,
        dir_subsampling="../saved_experiments_categorial_features/2024-10-03_DefaultUCI",
        name_subsampling_file="defaultUCI_sub_10_to_1",
        get_indexes=False,
    )
else:
   X,y = initial_X,initial_y
   
#output_dir_path =  "../saved_experiments_categorial_features/2024-12-16_feedzai"  
   
#output_dir_path =  "../saved_experiments_categorial_features/2024-10-03_DefaultUCI"  
#"defaultUCI_sub_original_to_10"
#"defaultUCI_sub_10_to_1"

#output_dir_path =  "../saved_experiments_categorial_features/2024-06-19_BankMarketing" 
#bankmarketing_sub_original_to_1

#"../saved_experiments_categorial_features/2024-06-19_BankChurners"
#'BankChurners_sub_original_to_1'

#"../saved_experiments_categorial_features/2024-06-19_TelcoChurn"
#'TelcoChurn_sub_original_to_10'
#'TelcoChurn_sub_10_to_1'
   
#dir_subsampling="../saved_experiments_categorial_features/2024-06-19_Census",
#name_subsampling_file="census_sub_original_to_10",



# all features
output_dir_path = "../saved_experiments_categorial_features/2024-12-16_feedzai/2025/lgbm/drfextra/"
Path(output_dir_path).mkdir(parents=True, exist_ok=True)
init_name_file_original = "2024-11-30-lgbm_"


###############################################################
########################### RUN ###############################
###############################################################
fun_tr_str = FunctionTransformer(to_str)
fun_tr_float = FunctionTransformer(to_float)
numeric_transformer = Pipeline(
    steps=[('Transform_float',fun_tr_float)]
)
categorical_transformer = Pipeline(steps=[('Transform_str',fun_tr_str),
                                          ("OneHot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                               ("cat", categorical_transformer, categorical_features),])

model = Pipeline(steps=[("preprocessor", preprocessor), ('rf', clf)])
balanced_model = Pipeline(steps=[("preprocessor", preprocessor), 
                                    ('rf',balanced_clf)])

# Initial run
for i in range(n_iter):
    list_oversampling_and_params = [
            (
                "MGS(mu)(d+1)(EmpCov) DRFskExtra classique (mtry=def=sqrt)",
                MultiOutPutClassifier_and_MGS(
                    K=K_MGS,
                    llambda=llambda_MGS,
                    categorical_features=categorical_features,
                    Classifier=DrfSkExtraClassifier(random_state=i,n_jobs=5, n_estimators=100),
                    random_state=0,
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
    
    splitter_stratified = StratifiedKFold(n_splits=5,shuffle=True,random_state=100+i)
    name_file = init_name_file_original + str(i) +'.npy'
    run_eval(output_dir=output_dir_path,name_file=name_file,X=X,y=y,
            list_oversampling_and_params=list_oversampling_and_params,
            splitter=splitter_stratified,
            categorical_features=categorical_features)
    print('FIN Iteration :',i)
    

##############################################################################################################################
list_oversampling_and_params = [
            ("None", NoSampling(), {}, model),
            ("CW", NoSampling(), {}, balanced_model),
            ('ROS', RandomOverSampler(sampling_strategy="minority",random_state=i),{}, model),
            ('RUS', RandomUnderSampler(sampling_strategy="majority",replacement=False,random_state=i),{}, model),
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
                    kind_cov = 'EmpCov',
                    mucentered=True,
                    fit_nn_on_continuous_only= True,
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
                    kind_cov = 'EmpCov',
                    mucentered=True,
                    fit_nn_on_continuous_only= True,
                ),
                {},
                model,
            ),
            ('MGS-NC(mu)(d+1)(EmpCov)',WMGS_NC_cov(K=K_MGS,llambda=llambda_MGS,kind_cov = 'EmpCov',mucentered=True,version=1,categorical_features=categorical_features,random_state=i),{},model),
            ('MGS-NC(mu)(d+1)(IdCov)',WMGS_NC_cov(K=K_MGS,llambda=llambda_MGS,kind_cov = 'IdCov',mucentered=True,version=1,categorical_features=categorical_features,random_state=i),{},model),
            (
                "MGS(mu)(d+1)(EmpCov) RF classique",
                MultiOutPutClassifier_and_MGS(
                    K=K_MGS,
                    llambda=llambda_MGS,
                    categorical_features=categorical_features,
                    Classifier=RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=5),
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
                "MGS(mu)(d+1)(EmpCov) DRFsk classique",
                MultiOutPutClassifier_and_MGS(
                    K=K_MGS,
                    llambda=llambda_MGS,
                    categorical_features=categorical_features,
                    Classifier=DrfSk(random_state=0,n_jobs=5),
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
                "MGS(mu)(d+1)(EmpCov) DRFskExtra classique (mtry=1.0)",
                MultiOutPutClassifier_and_MGS(
                    K=K_MGS,
                    llambda=llambda_MGS,
                    categorical_features=categorical_features,
                    Classifier=DrfSkExtraClassifier(random_state=i,n_jobs=5, n_estimators=100, max_features=1.0),
                    random_state=0,
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
            (
                "MGS(mu)(d+1)(EmpCov) RFstr",
                MultiOutPutClassifier_and_MGS(
                    K=K_MGS,
                    llambda=llambda_MGS,
                    categorical_features=categorical_features,
                    Classifier=RandomForestClassifier(n_estimators=100,random_state=0,n_jobs=5),
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
                    to_encode=False,
                    to_encode_onehot=False,
                    bool_rf=False,
                    bool_rf_str=True,
                    bool_rf_regressor=False,
                    bool_drf=False,
                    fit_nn_on_continuous_only= True,
                ),
                {},
                model,
            ),
            (
                "MGS(mu)(d+1)(EmpCov) RFc",
                MultiOutPutClassifier_and_MGS(
                    K=K_MGS,
                    llambda=llambda_MGS,
                    categorical_features=categorical_features,
                    Classifier=RandomForestClassifier(n_estimators=100,criterion='gini',random_state=0,n_jobs=5),
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
                    to_encode=False,
                    to_encode_onehot=True,
                    bool_rf=True,
                    bool_drf=False,
                    fit_nn_on_continuous_only= True,
                ),
                {},
                model,
            ),
            (
                "MGS(mu)(d+1)(EmpCov) RFr",
                MultiOutPutClassifier_and_MGS(
                    K=K_MGS,
                    llambda=llambda_MGS,
                    categorical_features=categorical_features,
                    Classifier=RandomForestRegressor(n_estimators=100,random_state=0,n_jobs=5),
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
                    to_encode=False,
                    to_encode_onehot=True,
                    bool_rf_regressor=True,
                    bool_drf=False,
                    fit_nn_on_continuous_only= True,
                ),
                {},
                model,
            ),
            (
                "MGS(mu)(d+1)(EmpCov) DRFsk",
                MultiOutPutClassifier_and_MGS(
                    K=K_MGS,
                    llambda=llambda_MGS,
                    categorical_features=categorical_features,
                    Classifier=DrfSkRegressor(random_state=0,n_jobs=5),
                    random_state=i,
                    kind_cov = 'EmpCov',
                    mucentered=True,
                    to_encode=False,
                    to_encode_onehot=True,
                    bool_rf=False,
                    bool_rf_regressor=False,
                   bool_drfsk_regressor=True,
                    bool_drf=False,
                    fit_nn_on_continuous_only= True,
                ),
                {},
                model,
            ),
            
        ]






list_oversampling_and_params = [
        ("None", NoSampling(), {}, model),
        (
            "MGS(mu)(d+1)(EmpCov) DRFsk classique (mtry=def=sqrt)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSk(random_state=0,n_jobs=5),
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
            "MGS(mu)(d+1)(EmpCov) DRFsk classique (mtry=0.5)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSk(random_state=0,n_jobs=5,max_features=0.5),
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
            "MGS(mu)(d+1)(EmpCov) DRFsk classique (mtry=1.0)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSk(random_state=0,n_jobs=5,max_features=1.0),
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
            "MGS(mu)(d+1)(EmpCov) DRFskExtra classique (mtry=def=sqrt)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSkExtraClassifier(random_state=i,n_jobs=5, n_estimators=100),
                random_state=0,
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
            "MGS(mu)(d+1)(EmpCov) DRFskExtra classique (mtry=0.5)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSkExtraClassifier(random_state=i,n_jobs=5, n_estimators=100, max_features=0.5),
                random_state=0,
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
            "MGS(mu)(d+1)(EmpCov) DRFskExtra classique (mtry=1.0)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSkExtraClassifier(random_state=i,n_jobs=5, n_estimators=100, max_features=1.0),
                random_state=0,
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





list_oversampling_and_params = [
        ("None", NoSampling(), {}, model),
        (
            "MGS(mu)(d+1)(EmpCov) DRFsk reg (mtry=def=1.0)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSkRegressor(random_state=0,n_jobs=5),
                random_state=i,
                kind_cov = 'EmpCov',
                mucentered=True,
                to_encode=False,
                to_encode_onehot=True,
                bool_rf=False,
                bool_rf_regressor=False,
                bool_drfsk_regressor=True,
                bool_drf=False,
                fit_nn_on_continuous_only= True,
            ),
            {},
            model,
        ),
        (
            "MGS(mu)(d+1)(EmpCov) DRFsk reg (mtry=0.5)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSkRegressor(random_state=0,n_jobs=5,max_features=0.5),
                random_state=i,
                kind_cov = 'EmpCov',
                mucentered=True,
                to_encode=False,
                to_encode_onehot=True,
                bool_rf=False,
                bool_rf_regressor=False,
                bool_drfsk_regressor=True,
                bool_drf=False,
                fit_nn_on_continuous_only= True,
            ),
            {},
            model,
        ),
        (
            "MGS(mu)(d+1)(EmpCov) DRFsk reg (mtry=sqrt)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                llambda=llambda_MGS,
                categorical_features=categorical_features,
                Classifier=DrfSkRegressor(random_state=0,n_jobs=5,max_features="sqrt"),
                random_state=i,
                kind_cov = 'EmpCov',
                mucentered=True,
                to_encode=False,
                to_encode_onehot=True,
                bool_rf=False,
                bool_rf_regressor=False,
                bool_drfsk_regressor=True,
                bool_drf=False,
                fit_nn_on_continuous_only= True,
            ),
            {},
            model,
        ),
    ]


    
