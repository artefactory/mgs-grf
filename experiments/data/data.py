from collections import Counter
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

from experiments.validation.classif_experiments import subsample_to_ratio_indices
from ucimlrepo import fetch_ucirepo


DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "experiments", "data", "externals"
)
output_dir_path = "../saved_experiments_categorial_features/BankChurners_example"
Path(output_dir_path).mkdir(parents=True, exist_ok=True)


def load_BankChurners_data():
    r"""
    Load BankChurners data set from data\dexternals folder
    The name of the file shoulde be : BankChurners.csv
    """
    filename = "BankChurners.csv"
    df_bankchurners = pd.read_csv(os.path.join(DATA_DIR, filename))

    X_bankChurners = df_bankchurners.drop(
        [
            "CLIENTNUM",
            "Attrition_Flag",
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
        ],
        axis=1,
    ).to_numpy()
    y_bankChurners = (
        df_bankchurners[["Attrition_Flag"]]
        .replace({"Attrition_Flag": {"Existing Customer": 0, "Attrited Customer": 1}})
        .to_numpy()
        .ravel()
    )
    return X_bankChurners, y_bankChurners


def load_BankMarketing_data():
    """
    Load BankChurners data set from UCI Irvine.
    """
    # fetch dataset
    bank_marketing = fetch_ucirepo(id=222)
    # data (as pandas dataframes)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    X.fillna("unknow", inplace=True)  # fillna
    y.replace({"y": {"yes": 1, "no": 0}}, inplace=True)
    X = X.to_numpy()
    y = y.to_numpy().ravel()
    return X, y


def load_BankChurners_data_():
    r"""
    Load BankChurners data set from data\dexternals folder AND subsample it to 1% imbalance ratio
    The name of the file shoulde be : BankChurners.csv
    """
    filename = "BankChurners.csv"
    df_bankchurners = pd.read_csv(os.path.join(DATA_DIR, filename))

    X_bankChurners = df_bankchurners.drop(
        [
            "CLIENTNUM",
            "Attrition_Flag",
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
        ],
        axis=1,
    ).to_numpy()
    pd.set_option("future.no_silent_downcasting", True)
    y_bankChurners = (
        df_bankchurners[["Attrition_Flag"]]
        .replace({"Attrition_Flag": {"Existing Customer": 0, "Attrited Customer": 1}})
        .to_numpy()
        .ravel()
        .astype(int)
    )
    indices_kept_1 = subsample_to_ratio_indices(
        X=X_bankChurners,
        y=y_bankChurners,
        ratio=0.01,
        seed_sub=5,
        output_dir_subsampling=output_dir_path,
        name_subsampling_file="BankChurners_sub_original_to_1",
    )
    X, y = X_bankChurners[indices_kept_1], y_bankChurners[indices_kept_1]
    return X, y
