from collections import Counter
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

DATA_DIR = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data", "externals")


def load_BankChurners_data():
    r"""
    Load BankChurners data set from data\dexternals folder
    The name of the file shoulde be : BankChurners.csv
    """
    filename = "BankChurners.csv"
    df_bankchurners = pd.read_csv(
        os.path.join(DATA_DIR, filename)
    )

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


def load_defaultUCI_data():
    # fetch dataset
    default_of_credit_card_clients = fetch_ucirepo(id=350)

    # data (as pandas dataframes)
    X = default_of_credit_card_clients.data.features
    y = default_of_credit_card_clients.data.targets
    return X.to_numpy(), y.to_numpy().ravel()


def load_TelcoChurn_data():
    r"""
    Load TelcoChurn data set from data\dexternals folder
    The name of the file shoulde be : Telco-Customer-Churn.csv
    """
    filename = "Telco-Customer-Churn.csv"
    df_telco_churn = pd.read_csv(
        os.path.join(DATA_DIR, filename)
    )
    df_telco_churn.replace({"TotalCharges": {" ": "0"}}, inplace=True)
    df_telco_churn[["TotalCharges"]] = df_telco_churn[["TotalCharges"]].astype(float)

    X_telco = df_telco_churn.drop(["customerID", "Churn"], axis=1).to_numpy()
    y_telco = df_telco_churn[["Churn"]].replace({"Churn": {"No": 0, "Yes": 1}}).to_numpy().ravel()

    return X_telco, y_telco


def load_census_data():
    """
    Load BankChurners data set from UCI Irvine.
    """
    # fetch dataset
    census_income = fetch_ucirepo(id=20)
    # data (as pandas dataframes)
    X = census_income.data.features
    y = census_income.data.targets

    X.fillna("unknow", inplace=True)  # fillna
    y.replace({"income": {">50K": 1, ">50K.": 1, "<=50K": 0, "<=50K.": 0}}, inplace=True)
    X = X.to_numpy()
    y = y.to_numpy().ravel()
    return X, y


def load_feedzai_data():
    r"""
    Load Base data set from data\dexternals folder
    The name of the file shoulde be : Base.csv
    """
    filename = "Base.csv"
    try:
        df_feedzai = pd.read_csv(os.path.join(DATA_DIR, filename))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Pima dataset not found. It must be downloaded and
                                placed in the folder {DATA_DIR} under the name {filename}"""
        )

    X_feedzai = df_feedzai.drop(["fraud_bool"], axis=1).to_numpy()
    y_feedzai = df_feedzai[["fraud_bool"]].to_numpy().ravel()  # be consistent with X
    return X_feedzai, y_feedzai


def decode_one_hot(row, columns):
    """
    Parameters
    ----------
    row : pd.DataFrame instance with shape[0]=1
    columns : list
    ----------
    Return the elment instance c of columns for which row[c]=1.
    """
    for c in columns:
        if row[c] == 1:
            return c


def load_covertype_data(
    dict_mapping={1: 0, 4: 1},
):  # {1:0, 2: 0, 3:0, 4:0, 5:0, 6:0 ,7:0 ,8:0}
    """
    Load Covertype data set from UCI Irvine.
    """
    covertype = fetch_ucirepo(id=31)  # fetch dataset
    original_X = covertype.data.features  # data (as pandas dataframes)
    original_y = covertype.data.targets
    X = original_X[
        [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
    ].copy()

    columns_soil = [
        "Soil_Type1",
        "Soil_Type2",
        "Soil_Type3",
        "Soil_Type4",
        "Soil_Type5",
        "Soil_Type6",
        "Soil_Type7",
        "Soil_Type8",
        "Soil_Type9",
        "Soil_Type10",
        "Soil_Type11",
        "Soil_Type12",
        "Soil_Type13",
        "Soil_Type14",
        "Soil_Type15",
        "Soil_Type16",
        "Soil_Type17",
        "Soil_Type18",
        "Soil_Type19",
        "Soil_Type20",
        "Soil_Type21",
        "Soil_Type22",
        "Soil_Type23",
        "Soil_Type24",
        "Soil_Type25",
        "Soil_Type26",
        "Soil_Type27",
        "Soil_Type28",
        "Soil_Type29",
        "Soil_Type30",
        "Soil_Type31",
        "Soil_Type32",
        "Soil_Type33",
        "Soil_Type34",
        "Soil_Type35",
        "Soil_Type36",
        "Soil_Type37",
        "Soil_Type38",
        "Soil_Type39",
        "Soil_Type40",
    ]
    series_soil = original_X[columns_soil].apply(decode_one_hot, columns=columns_soil, axis=1)
    X[["Soil_Type"]] = series_soil.to_frame()

    columns_wilderness = [
        "Wilderness_Area1",
        "Wilderness_Area2",
        "Wilderness_Area3",
        "Wilderness_Area4",
    ]
    series_wilderness = original_X[columns_wilderness].apply(
        decode_one_hot, columns=columns_wilderness, axis=1
    )
    X[["Wilderness_Area"]] = series_wilderness.to_frame()

    if dict_mapping is not None:
        df = pd.concat([X, original_y], axis=1)
        df = df[df["Cover_Type"].isin([int(key) for key in dict_mapping.keys()])].copy()
        df.replace({"Cover_Type": dict_mapping}, inplace=True)
        X = df.drop(["Cover_Type"], axis=1).to_numpy()
        y = df[["Cover_Type"]].to_numpy().ravel()
        return X, y
    else:
        return X.to_numpy(), original_y.to_numpy().ravel()
    

output_dir_path = "../saved_experiments_categorial_features/BankChurners_example"
Path(output_dir_path).mkdir(parents=True, exist_ok=True)
from validation.classif_experiments import subsample_to_ratio_indices
def load_BankChurners_data_():
    r"""
    Load BankChurners data set from data\dexternals folder AND subsample it to 1% imbalance ratio
    The name of the file shoulde be : BankChurners.csv
    """
    filename = "BankChurners.csv"
    df_bankchurners = pd.read_csv(
        os.path.join(DATA_DIR, filename)
    )

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
    indices_kept_1 = subsample_to_ratio_indices(X=X_bankChurners,y=y_bankChurners,ratio=0.01,seed_sub=5,output_dir_subsampling=output_dir_path,name_subsampling_file='BankChurners_sub_original_to_1')
    X, y = X_bankChurners[indices_kept_1],y_bankChurners[indices_kept_1]
    return X, y
