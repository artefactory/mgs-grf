import os
import sys

sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from drf import drf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb

from oversampling_strategies.oversampling_strategies import (
    SMOTE_cat,
    MGS_cat,
    MGS_NC,
    MultiOutPutClassifier_and_MGS,
    NoSampling,
)
from validation.classif_experiments import run_eval, read_subsampling_indices
from data.data import load_covertype_data

##################################################
##################################################
X, y = load_covertype_data()
numeric_features = [0,1,2,3,4,5,6,7,8,9]
categorical_features = [10,11]
K_MGS = max(len(numeric_features) + 1, 5)
print('Value K_MGS : ', K_MGS)


clf = lgb.LGBMClassifier(verbosity=-1,n_jobs=5,random_state=0)
balanced_clf = lgb.LGBMClassifier(class_weight='balanced',verbosity=-1,n_jobs=5,random_state=0)
#clf = RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs=5)
#balanced_clf = RandomForestClassifier(n_estimators=100,criterion='gini',class_weight='balanced',n_jobs=5)
n_iter = 20

output_dir_path_original = "../saved_experiments_categorial_features/2024-07-30_Covertype/lgbm/kind_generation_cat_features/original"
Path(output_dir_path_original).mkdir(parents=True, exist_ok=True)
init_name_file_original = "2024-07-30_Covertype_depthNone_"

if False :
    X_10, y_10 = read_subsampling_indices(
        X=X,
        y=y,
        dir_subsampling="../saved_experiments_categorial_features/2024-06-19_TelcoChurn",
        name_subsampling_file="TelcoChurn_sub_original_to_10",
        get_indexes=False,
    )
    output_dir_path_subsampled_10 = "../saved_experiments_categorial_features/2024-06-19_TelcoChurn/lgbm/kind_generation_cat_features/subsample_to_10"
    Path(output_dir_path_subsampled_10).mkdir(parents=True, exist_ok=True)
    init_name_file_subsampled = "2024-06-19_TelcoChurn_depthNone_"

    X_1, y_1 = read_subsampling_indices(
        X=X,
        y=y,
        dir_subsampling="../saved_experiments_categorial_features/2024-06-19_TelcoChurn",
        name_subsampling_file="TelcoChurn_sub_10_to_1",
        get_indexes=False,
    )
    output_dir_path_subsampled = "../saved_experiments_categorial_features/2024-06-19_TelcoChurn/lgbm/kind_generation_cat_features/subsample_to_1"
    Path(output_dir_path_subsampled).mkdir(parents=True, exist_ok=True)
    init_name_file_subsampled = "2024-06-19_TelcoChurn_depthNone_"


##################################################
##################################################
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

for i in range(n_iter):
    list_oversampling_and_params = [
        ("None", NoSampling(), {}, model),
        ("c_weight", NoSampling(), {}, balanced_model),
        (
            "RUS",
            RandomUnderSampler(
                sampling_strategy="majority", replacement=False, random_state=i
            ),
            {},
            model,
        ),
        (
            "SMOTE (K=5) dist without discrete feat (sampling rand)",
            SMOTE_cat(
                K=5, categorical_features=categorical_features, version=2, random_state=i
            ),
            {},
            model,
        ),
        (
            "MGS NC (d+1)(sampling rand)",
            MGS_NC(
                K=K_MGS,
                n_points=K_MGS,
                llambda=1.0,
                categorical_features=categorical_features,
                version=2,
                random_state=i,
            ),
            {},
            model,
        ),
        (
            "MGS (d+1) dist without discrete feat (sampling rand)",
            MGS_cat(
                K=K_MGS,
                n_points=K_MGS,
                llambda=1.0,
                categorical_features=categorical_features,
                version=2,
                random_state=i,
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
            "SMOTE (K=5) dist without discrete feat (inv dist)",
            SMOTE_cat(
                K=5, categorical_features=categorical_features, version=3, random_state=i
            ),
            {},
            model,
        ),
        (
            "MGS NC (d+1) (inv dist)",
            MGS_NC(
                K=K_MGS,
                n_points=K_MGS,
                llambda=1.0,
                categorical_features=categorical_features,
                version=3,
                random_state=i,
            ),
            {},
            model,
        ),
        (
            "MGS (d+1) dist without discrete feat (inv dist)",
            MGS_cat(
                K=K_MGS,
                n_points=K_MGS,
                llambda=1.0,
                categorical_features=categorical_features,
                version=3,
                random_state=i,
            ),
            {},
            model,
        ),
        (
            "5-NN MGS(d+1)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                categorical_features=categorical_features,
                Classifier=KNeighborsClassifier(n_neighbors=5),
                random_state=i,
            ),
            {},
            model,
        ),
        (
            "DRF(default) MGS(d+1)",
            MultiOutPutClassifier_and_MGS(
                K=K_MGS,
                categorical_features=categorical_features,
                Classifier=drf(min_node_size=1, num_trees=100, splitting_rule="FourierMMD"),
                random_state=i,
                to_encode=True,
            ),
            {},
            model,
        ),
    ]
####### original
    splitter_stratified = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=100 + i
    )
    name_file = init_name_file_original + str(i) + ".npy"
    run_eval(
        output_dir=output_dir_path_original,
        name_file=name_file,
        X=X,
        y=y,
        list_oversampling_and_params=list_oversampling_and_params,
        splitter=splitter_stratified,
        categorical_features=categorical_features,
    )


if False:
##### Subsampled
    splitter_stratified = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=100 + i
    )
    name_file = init_name_file_subsampled + str(i) + ".npy"
    run_eval(
        output_dir=output_dir_path_subsampled_10,
        name_file=name_file,
        X=X_10,
        y=y_10,
        list_oversampling_and_params=list_oversampling_and_params,
        splitter=splitter_stratified,
        categorical_features=categorical_features,
    )
    
##### Subsampled
    splitter_stratified = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=100 + i
    )
    name_file = init_name_file_subsampled + str(i) + ".npy"
    run_eval(
        output_dir=output_dir_path_subsampled,
        name_file=name_file,
        X=X_1,
        y=y_1,
        list_oversampling_and_params=list_oversampling_and_params,
        splitter=splitter_stratified,
        categorical_features=categorical_features,
    )



