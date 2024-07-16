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

from oversampling_strategies.oversampling_strategies import (
    SMOTE_cat,
    MGS_cat,
    MGS_NC,
    MultiOutPutClassifier_and_MGS,
    NoSampling,
)
from validation.classif_experiments import run_eval, read_subsampling_indices
from data.data import load_BankChurners_data

##################################################
##################################################
numeric_features = [0, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
categorical_features = [1, 3, 4, 5, 6]
X, y = load_BankChurners_data()
# output_dir_path =  "../saved_experiments_categorial_features/2024-06-19_BankMarketing/kind_generation_cat_features" ## Fill it
# Path(output_dir_path).mkdir(parents=True, exist_ok=True)
clf = RandomForestClassifier(
    n_estimators=100, criterion="gini", n_jobs=5, random_state=0
)
balanced_clf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    class_weight="balanced",
    n_jobs=5,
    random_state=0,
)
n_iter = 20

output_dir_path_original = "../saved_experiments_categorial_features/2024-06-19_BankChurners/kind_generation_cat_features/original"
Path(output_dir_path_original).mkdir(parents=True, exist_ok=True)
init_name_file_original = "2024-06-19-RF100_BankChurners_depthNone_"

X_1, y_1 = read_subsampling_indices(
    X=X,
    y=y,
    dir_subsampling="../saved_experiments_categorial_features/2024-06-19_BankChurners",
    name_subsampling_file="BankChurners_sub_original_to_1",
    get_indexes=False,
)
output_dir_path_subsampled = "../saved_experiments_categorial_features/2024-06-19_BankChurners/kind_generation_cat_features/subsample_to_1"
Path(output_dir_path_subsampled).mkdir(parents=True, exist_ok=True)
init_name_file_subsampled = "2024-06-19-RF100_BankChurners_sub1_depthNone_"


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

list_oversampling_and_params = [
    ("None", NoSampling(), {}, model),
    ("c_weight", NoSampling(), {}, balanced_model),
    (
        "RUS",
        RandomUnderSampler(
            sampling_strategy="majority", replacement=False, random_state=0
        ),
        {},
        model,
    ),
    (
        "SMOTE (K=5) dist without discrete feat (sampling rand)",
        SMOTE_cat(
            K=5, categorical_features=categorical_features, version=2, random_state=0
        ),
        {},
        model,
    ),
    (
        "MGS NC K= 7(sampling rand)",
        MGS_NC(
            K=7,
            n_points=7,
            llambda=1.0,
            categorical_features=categorical_features,
            version=2,
            random_state=0,
        ),
        {},
        model,
    ),
    (
        "MGS K=7 dist without discrete feat (sampling rand)",
        MGS_cat(
            K=7,
            n_points=7,
            llambda=1.0,
            categorical_features=categorical_features,
            version=2,
            random_state=0,
        ),
        {},
        model,
    ),
    (
        "SmoteNC (K=5)",
        SMOTENC(
            k_neighbors=5, categorical_features=categorical_features, random_state=0
        ),
        {},
        model,
    ),
    (
        "SMOTE (K=5) dist without discrete feat (inv dist)",
        SMOTE_cat(
            K=5, categorical_features=categorical_features, version=3, random_state=0
        ),
        {},
        model,
    ),
    (
        "MGS NC K=7 (inv dist)",
        MGS_NC(
            K=7,
            n_points=7,
            llambda=1.0,
            categorical_features=categorical_features,
            version=3,
            random_state=0,
        ),
        {},
        model,
    ),
    (
        "MGS K=7 dist without discrete feat (inv dist)",
        MGS_cat(
            K=7,
            n_points=7,
            llambda=1.0,
            categorical_features=categorical_features,
            version=3,
            random_state=0,
        ),
        {},
        model,
    ),
    (
        "5-NN MGS(K=7)",
        MultiOutPutClassifier_and_MGS(
            K=7,
            categorical_features=categorical_features,
            Classifier=KNeighborsClassifier(n_neighbors=5),
            random_state=0,
        ),
        {},
        model,
    ),
    (
        "DRF(default) MGS(K=7)",
        MultiOutPutClassifier_and_MGS(
            K=7,
            categorical_features=categorical_features,
            Classifier=drf(min_node_size=1, num_trees=100, splitting_rule="FourierMMD"),
            random_state=0,
            to_encode=True,
        ),
        {},
        model,
    ),
]

####### original
for i in range(n_iter):
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

##### Subsampled
for i in range(n_iter):
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
