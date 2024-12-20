import os

import pandas as pd
import numpy as np
from collections import Counter
from scipy.io.arff import loadarff

DATA_DIR = os.path.join(
    os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data", "externals"
)


def load_abalone_data():
    """
    Loads Abalone data set from data\dexternals folder.
    The name of the file shoulde be : abalone.data
    """
    filename = "abalone.data"
    try:
        df_abalone = pd.read_csv(
            os.path.join(DATA_DIR, filename),
            names=[
                "Sex",
                "Length",
                "Diameter",
                "Height",
                "Whole_weight",
                "Shucked_weight",
                "Viscera_weight",
                "Shell_weight",
                "Rings",
            ],
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Abalone dataset not found. It must be downloaded and
                                placed in the folder {DATA_DIR} under the name {filename}"""
        )
    ### the abalone with a number of Rings equall to 18 are the samples of the minority class
    dict_mapping = {18: 1}
    for key in [
        15,
        7,
        9,
        10,
        8,
        20,
        16,
        19,
        14,
        11,
        12,
        13,
        5,
        4,
        6,
        21,
        17,
        22,
        1,
        3,
        26,
        23,
        29,
        2,
        27,
        25,
        24,
    ]:
        dict_mapping[key] = 0

    df_abalone = df_abalone.replace({"Rings": dict_mapping})

    X_abalone = df_abalone.drop(["Rings", "Sex"], axis=1).to_numpy()
    y_abalone = df_abalone[["Rings"]].values.ravel()
    return X_abalone, y_abalone


def load_pima_data():
    """
    Load PIMA diabates data set from data\dexternals folder
    The name of the file shoulde be : diabetes.csv
    """
    filename = "diabetes.csv"
    try:
        df_diabete = pd.read_csv(os.path.join(DATA_DIR, filename))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Pima dataset not found. It must be downloaded and
                                placed in the folder {DATA_DIR} under the name {filename}"""
        )

    X_pima = df_diabete.drop(["Outcome"], axis=1).to_numpy()
    y_pima = df_diabete[["Outcome"]].to_numpy().ravel()  # be consistent with X
    return X_pima, y_pima


def load_phoneme_data():
    """
    Load Phoneme diabates data set from data\dexternals folder
    The name of the file shoulde be : phoneme.csv
    """
    filename = "phoneme.csv"
    try:
        df_phoneme = pd.read_csv(
            os.path.join(DATA_DIR, filename),
            names=["Aa", " Ao", " Dcl", " Iy", " Sh", " Class"],
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Pima dataset not found. It must be downloaded and
                                placed in the folder {DATA_DIR} under the name {filename}"""
        )

    X_phoneme = df_phoneme.drop([" Class"], axis=1).to_numpy()
    y_phoneme = df_phoneme[[" Class"]].to_numpy().ravel()
    return X_phoneme, y_phoneme


def load_df_phoneme_positifs():
    """
    Load Phoneme data set from data\dexternals folder and then keep only keep the minority (positive) samples
    The name of the file shoulde be : phoneme.csv
    """
    df_phoneme = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "phoneme.csv",
        ),
        names=["Aa", " Ao", " Dcl", " Iy", " Sh", " Class"],
    )
    df_phoneme_positifs = (
        df_phoneme[df_phoneme[" Class"] == 1].copy().reset_index(drop=True)
    )
    df_phoneme_positifs.drop(
        [" Class"], axis=1, inplace=True
    )  # personnally I prefer not to use inplace
    return df_phoneme_positifs


def load_yeast_data():
    """
    Load Yeast data set from data\dexternals folder
    The name of the file shoulde be : yeast.data
    """
    df_yeast = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "yeast.data",
        ),
        sep=r"\s+",
        names=[
            "Sequence_Name",
            "mcg",
            "gvh",
            "alm",
            "mit",
            "erl",
            "pox",
            "vac",
            "nuc",
            "localization_site",
        ],
    )
    df_yeast.replace(
        {
            "localization_site": {
                "MIT": 0,
                "NUC": 0,
                "CYT": 0,
                "ME1": 0,
                "EXC": 0,
                "ME2": 0,
                "VAC": 0,
                "POX": 0,
                "ERL": 0,
                "ME3": 1,
            }
        },
        inplace=True,
    )

    X_yeast = df_yeast.drop(["Sequence_Name", "localization_site"], axis=1).to_numpy()
    y_yeast = df_yeast[["localization_site"]].to_numpy().ravel()
    return X_yeast, y_yeast


def load_haberman_data():
    """
    Load Haberman data set from data\dexternals folder
    The name of the file shoulde be : haberman.data
    """
    df_haberman = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "haberman.data",
        ),
        sep=",",
        header=None,
    )
    df_haberman.columns = ["Age", "year_Op", "npand", "Class"]
    df_haberman.replace({"Class": {1: 0, 2: 1}}, inplace=True)
    X_haberman = df_haberman.drop(["Class"], axis=1).to_numpy()
    y_haberman = df_haberman[["Class"]].to_numpy().ravel()
    return X_haberman, y_haberman


def load_magictel_data():
    """
    Load haberman data set from data\dexternals folder
    The name of the file shoulde be : magictel.arff
    """
    raw_magic = loadarff(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "magictelescope.arff",
        )
    )
    df_magic = pd.DataFrame(raw_magic[0])
    df_magic.replace({"class": {b"h": 0, b"g": 1}}, inplace=True)
    X_magic = df_magic.drop(["class"], axis=1).to_numpy()
    y_magic = df_magic[["class"]].to_numpy().ravel()

    return X_magic, y_magic


def load_california_data():
    """
    Load California data set from data\dexternals folder
    The name of the file shoulde be : california.arff
    """
    raw_cal_housing = loadarff(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "california.arff",
        )
    )
    df_cal_housing = pd.DataFrame(raw_cal_housing[0])
    df_cal_housing.replace({"price": {b"True": 1, b"False": 0}}, inplace=True)
    X_cal_housing = df_cal_housing.drop(["price"], axis=1).to_numpy()
    y_cal_housing = df_cal_housing[["price"]].to_numpy().ravel()

    return X_cal_housing, y_cal_housing


def load_house_data():
    """
    Load House_16h data set from data\dexternals folder
    The name of the file shoulde be : house_16H.arff
    """
    raw_house_16H = loadarff(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "house_16H.arff",
        )
    )
    df_house_16H = pd.DataFrame(raw_house_16H[0])
    df_house_16H.replace({"binaryClass": {b"P": 0, b"N": 1}}, inplace=True)
    X_house_16H = df_house_16H.drop(["binaryClass"], axis=1).to_numpy()
    y_house_16H = df_house_16H[["binaryClass"]].to_numpy().ravel()

    return X_house_16H, y_house_16H


def load_vehicle_data():
    """
    Load vehicle data set from data\dexternals folder
    All the file dowloaded/unziped from UCI should be in a folder named vehicle vehicle
    """
    names_vehicle = [
        "COMPACTNESS",
        "CIRCULARITY",
        "DISTANCE_CIRCULARITY",
        "RADIUS_RATIO",
        "PR_AXIS_ASPECT_RATIO",
        "MAX_LENGTH_ASPECT_RATIO",
        "SCATTER_RATIO",
        "ELONGATEDNESS",
        "PR_AXIS_RECTANGULARITY",
        "MAX_LENGTH_RECTANGULARITY",
        "SCALED_VARIANCE_ALONG_MAJOR_AXIS",
        "SCALED_VARIANCE_ALONG_MINOR_AXIS",
        "SCALED_RADIUS_OF_GYRATION",
        "SKEWNESS_ABOUT_MAJOR_AXIS",
        "SKEWNESS_ABOUT_MINOR_AXIS",
        "KURTOSIS_ABOUT_MINOR_AXIS",
        "KURTOSIS_ABOUT_MAJOR_AXIS",
        "HOLLOWS_RATIO",
        "Class",
    ]
    df_tmp_1 = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "vehicle",
            "xaa.dat",
        ),
        sep=r"\s+",
        header=None,
    )
    df_tmp_2 = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "vehicle",
            "xai.dat",
        ),
        sep=r"\s+",
        header=None,
    )
    df_tmp_3 = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "vehicle",
            "xah.dat",
        ),
        sep=r"\s+",
        header=None,
    )
    df_tmp_4 = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "vehicle",
            "xag.dat",
        ),
        sep=r"\s+",
        header=None,
    )
    df_tmp_5 = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "vehicle",
            "xaf.dat",
        ),
        sep=r"\s+",
        header=None,
    )
    df_tmp_6 = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "vehicle",
            "xae.dat",
        ),
        sep=r"\s+",
        header=None,
    )
    df_tmp_7 = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "vehicle",
            "xad.dat",
        ),
        sep=r"\s+",
        header=None,
    )
    df_tmp_8 = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "vehicle",
            "xac.dat",
        ),
        sep=r"\s+",
        header=None,
    )
    df_tmp_9 = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "vehicle",
            "xab.dat",
        ),
        sep=r"\s+",
        header=None,
    )
    df_vehicle = pd.concat(
        [
            df_tmp_1,
            df_tmp_2,
            df_tmp_3,
            df_tmp_4,
            df_tmp_5,
            df_tmp_6,
            df_tmp_7,
            df_tmp_8,
            df_tmp_9,
        ],
        axis=0,
    )
    df_vehicle.columns = names_vehicle
    df_vehicle.replace(
        {"Class": {"van": 1, "saab": 0, "bus": 0, "opel": 0}}, inplace=True
    )
    X_vehicle = df_vehicle.drop(["Class"], axis=1).to_numpy()
    y_vehicle = df_vehicle[["Class"]].to_numpy().ravel()

    return X_vehicle, y_vehicle


def load_breastcancer_data():
    """
    Load Breast cancer wisconsin data set from data\dexternals folder
    The name of the file shoulde be : breast-cancer-wisconsin.data
    """
    df_b_cancer_w = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "breast-cancer-wisconsin.data",
        ),
        sep=",",
        header=None,
    )
    df_b_cancer_w = pd.read_csv(
        "/home/abdoulaye_sakho/Code/data/breast-cancer-wisconsin.data",
        sep=",",
        header=None,
    )
    df_b_cancer_w.columns = [
        "Sample_code_number",
        "Clump_thickness",
        "Uniformity_of_cell_size",
        "Uniformity_of_cell_shape",
        "Marginal_adhesion",
        "Single_epithelial_cell_size",
        "Bare_nuclei",
        "Bland_chromatin",
        "Normal_nucleoli",
        "Mitoses",
        "Class",
    ]
    df_b_cancer_w.replace({"Class": {2: 0, 4: 1}}, inplace=True)
    df_b_cancer_w = df_b_cancer_w[df_b_cancer_w.Bare_nuclei != "?"]
    df_b_cancer_w = df_b_cancer_w.astype({"Bare_nuclei": "int64"})
    df_b_cancer_w.drop_duplicates(
        subset=["Sample_code_number"], keep="first", inplace=True
    )
    X_b_cancer_w = df_b_cancer_w.drop(["Class"], axis=1).to_numpy()
    y_b_cancer_w = df_b_cancer_w[["Class"]].to_numpy().ravel()

    return X_b_cancer_w, y_b_cancer_w


def load_ionosphere_data():
    """
    Load ionosphere data set from data\dexternals folder
    The name of the file shoulde be : ionosphere.data
    """
    df_ionosphere = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "ionosphere.data",
        ),
        sep=",",
        header=None,
    )
    df_ionosphere.replace({34: {"g": 0, "b": 1}}, inplace=True)
    X_ionosphere = df_ionosphere.drop([1, 34], axis=1).to_numpy()
    y_ionosphere = df_ionosphere[[34]].to_numpy().ravel()

    return X_ionosphere, y_ionosphere


def load_credit_data():
    """
    Load Creditcard data set from data\dexternals folder
    The name of the file shoulde be : creditcard.csv
    """
    df_credit = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "creditcard.csv",
        )
    )
    meta_df_credit = df_credit[["Time"]]
    X_credit = df_credit.drop(["Time", "Class"], axis=1).to_numpy()
    y_credit = df_credit[["Class"]].to_numpy().ravel()

    return X_credit, y_credit, meta_df_credit


from ucimlrepo import fetch_ucirepo


def load_wine_data():
    """
    Load wine data set from ucimlrepo
    You should have installl ucimlrepo
    """
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets
    df_wine = pd.concat([X, y], axis=1)

    dict_mapping = {5: 0, 6: 0, 8: 1}
    df_wine = df_wine[df_wine["quality"].isin([5, 6, 8])].copy()
    df_wine.replace({"quality": dict_mapping}, inplace=True)
    X_wine = df_wine.drop(["quality"], axis=1).to_numpy()
    y_wine = df_wine[["quality"]].to_numpy().ravel()
    return X_wine, y_wine


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


def load_BankChurners_data():
    """
    Load BankChurners data set from data\dexternals folder
    The name of the file shoulde be : BankChurners.csv
    """
    df_bankchurners = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "BankChurners.csv",
        )
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
    return X.to_numpy(),y.to_numpy().ravel()


def load_TelcoChurn_data():
    """
    Load TelcoChurn data set from data\dexternals folder
    The name of the file shoulde be : Telco-Customer-Churn.csv
    """
    df_telco_churn = pd.read_csv(
        os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
            "data",
            "externals",
            "Telco-Customer-Churn.csv",
        )
    )
    df_telco_churn.replace({"TotalCharges": {" ": "0"}}, inplace=True)
    df_telco_churn[["TotalCharges"]] = df_telco_churn[["TotalCharges"]].astype(float)

    X_telco = df_telco_churn.drop(["customerID", "Churn"], axis=1).to_numpy()
    y_telco = (
        df_telco_churn[["Churn"]]
        .replace({"Churn": {"No": 0, "Yes": 1}})
        .to_numpy()
        .ravel()
    )

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
    y.replace({"income": {'>50K': 1,'>50K.':1, '<=50K': 0, '<=50K.':0}}, inplace=True)
    X = X.to_numpy()
    y = y.to_numpy().ravel()
    return X, y

def load_feedzai_data():
    """
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


def decode_one_hot(row,columns):
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

def load_covertype_data(dict_mapping={1:0,4:1}): #{1:0, 2: 0, 3:0, 4:0, 5:0, 6:0 ,7:0 ,8:0}
    """
    Load Covertype data set from UCI Irvine.
    """
    covertype = fetch_ucirepo(id=31) # fetch dataset 
    original_X = covertype.data.features # data (as pandas dataframes) 
    original_y = covertype.data.targets 
    X = original_X[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
                    'Horizontal_Distance_To_Roadways','Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points']].copy()
    
    columns_soil = ['Soil_Type1','Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6','Soil_Type7', 'Soil_Type8',
                    'Soil_Type9', 'Soil_Type10', 'Soil_Type11','Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15','Soil_Type16',
                    'Soil_Type17', 'Soil_Type18', 'Soil_Type19','Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23','Soil_Type24',
                    'Soil_Type25', 'Soil_Type26', 'Soil_Type27','Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31','Soil_Type32',
                    'Soil_Type33', 'Soil_Type34', 'Soil_Type35','Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39','Soil_Type40']
    series_soil = original_X[columns_soil].apply(decode_one_hot,columns=columns_soil,axis=1)
    X[['Soil_Type']]=series_soil.to_frame()
    
    columns_wilderness = ['Wilderness_Area1','Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']
    series_wilderness= original_X[columns_wilderness].apply(decode_one_hot,columns=columns_wilderness,axis=1)
    X[['Wilderness_Area']] = series_wilderness.to_frame()

    if dict_mapping is not None:        
        df = pd.concat([X, original_y], axis=1)
        df = df[df["Cover_Type"].isin([int(key) for key in dict_mapping.keys()])].copy()
        df.replace({"Cover_Type": dict_mapping}, inplace=True)
        X = df.drop(["Cover_Type"], axis=1).to_numpy()
        y = df[["Cover_Type"]].to_numpy().ravel()
        return X,y
    else: 
        return X.to_numpy(),original_y.to_numpy().ravel()
    

############ SYNTHETIC #########
    
from sklearn.utils import shuffle
def subsample_to_ratio(X, y, ratio=0.02, seed_sub=0):
    X_positifs = X[y == 1]
    X_negatifs = X[y == 0]

    np.random.seed(seed=seed_sub)
    n_undersampling_sub = int(
        (ratio * len(X_negatifs)) / (1 - ratio)
    )  ## compute the number of sample to keep
    ##int() in order to have upper integer part
    idx = np.random.randint(len(X_positifs), size=n_undersampling_sub)

    X_positifs_selected = X_positifs[idx]
    y_positifs_selected = y[y == 1][idx]

    X_res = np.concatenate([X_negatifs, X_positifs_selected], axis=0)
    y_res = np.concatenate([y[y == 0], y_positifs_selected], axis=0)
    X_res, y_res = shuffle(X_res, y_res)
    return X_res, y_res
    
def my_log_reg(x,beta=np.array([-8,7,6]),intercept = -2): 
    #beta = np.array([-8,7,6])
    
    tmp = x.dot(beta)
    z = tmp + intercept # add intercept
    res = np.exp(z) / (1 + np.exp(z))
    return  res

def proba_to_label(y_pred_probas, treshold=0.5):  # apply_threshold ?
    return np.array(np.array(y_pred_probas) >= treshold, dtype=int)

import matplotlib.pyplot as plt
def generate_synthetic_features_logreg(X,index_informatives,list_modalities=['A','B'],beta=np.array([-8,7,6]),intercept=-2,treshold=0.5):
    res_log_reg = my_log_reg(X[:,index_informatives],beta=beta,intercept=intercept)
    #plt.hist(res_log_reg)
    #plt.xlim([-0.05,1.05])
    #plt.title(r'y_pred_proba histogram for dimension=%i',fontsize=10)
    #plt.show()

    #pred_logreg = proba_to_label(y_pred_probas=res_log_reg, treshold=treshold)
    #array_final = np.char.replace(pred_logreg.astype(str), '0',list_modalities[0])
    #array_final = np.char.replace(array_final.astype(str), '1',list_modalities[1])
    #return array_final, pred_logreg
    real_pred_log = np.random.binomial(n=1,p=res_log_reg)
    array_final = np.char.replace(real_pred_log.astype(str), '0',list_modalities[0])
    array_final = np.char.replace(array_final.astype(str), '1',list_modalities[1])
    return array_final,real_pred_log


def generate_initial_data_onecat(dimension,n_samples,random_state=24,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension-2):
        np.random.seed(seed=random_state+i+1)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
        
    ### Feature categorical
    feature_cat_uniform, feature_cat_uniform_numeric = generate_synthetic_features_logreg(X=Xf,
                                                                                      index_informatives=[0,1,2],
                                                                                      list_modalities=['C','D'],beta=np.array([-8,7,6]),intercept=-2)
    if verbose==0:
        print('Composition of categorical feature : ', Counter(feature_cat_uniform))
    X_final = np.hstack((Xf,feature_cat_uniform_numeric.reshape(-1,1)))
    target, target_numeric = generate_synthetic_features_logreg(X=X_final,index_informatives=[0,1,2,-1],list_modalities=['No','Yes'],beta=[4,-3,-3,3],intercept=-3.5)
    if verbose==0:
        print('Composition of the target ', Counter(target))
    return X_final, target, target_numeric


def generate_initial_data_onecat_normal(dimension,n_samples,random_state=24,verbose=0):
    np.random.seed(random_state)
    Xf=np.random.multivariate_normal(mean=np.zeros((dimension-1,)),cov=np.eye(dimension-1,dimension-1),size=n_samples)
    ### Feature categorical
    feature_cat_uniform, feature_cat_uniform_numeric = generate_synthetic_features_logreg(X=Xf,
                                                                                          index_informatives=[0,1,2],
                                                                                           list_modalities=['C','D'])
        
    X_final = np.hstack((Xf,feature_cat_uniform_numeric.reshape(-1,1)))
    target, target_numeric = generate_synthetic_features_logreg(X=X_final,index_informatives=[0,1,2,-1],list_modalities=['No','Yes'],beta=[-7,4,6,7],intercept=-11,treshold=0.5)
    if verbose==0:
        print('Composition of the target before subsampling ', Counter(target_numeric))
    X_final,target_numeric = subsample_to_ratio(X=X_final, y=target_numeric, ratio=0.08, seed_sub=random_state)
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(feature_cat_uniform))
    return X_final, target_numeric, target_numeric





def generate_synthetic_features_logreg_triple(X,index_informatives,list_modalities=['A','B','C'],
                                              beta1=np.array([-8,7,6]),beta2=np.array([8,-7,6]),beta3=np.array([8,7,-6])):
    res_log_reg1 = my_log_reg(X[:,index_informatives],beta=beta1)
    res_log_reg2 = my_log_reg(X[:,index_informatives],beta=beta2)
    res_log_reg3 = my_log_reg(X[:,index_informatives],beta=beta3)
    res_log_reg = np.hstack((res_log_reg1.reshape(-1,1),res_log_reg2.reshape(-1,1),res_log_reg3.reshape(-1,1)))
    array_argmax = np.argmax(res_log_reg,axis=1)

    array_final = np.char.replace(array_argmax.astype(str), '0',list_modalities[0])
    array_final = np.char.replace(array_final.astype(str), '1',list_modalities[1])
    array_final = np.char.replace(array_final.astype(str), '2',list_modalities[2])
    return array_final,array_argmax

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
def generate_synthetic_features_logreg_2(X,index_informatives,list_modalities=['A','B'],beta=np.array([-8,7,6]),intercept=-2):
    res_log_reg = my_log_reg(X[:,index_informatives],beta=beta,intercept=-2)
    return res_log_reg
    
def generate_initial_data_twocat_fit(dimension,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension-3):
        np.random.seed(seed=random_state+i+1)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
        
    feature_cat_uniform, feature_cat_uniform_numeric = generate_synthetic_features_logreg_triple(X=Xf,index_informatives=[0,1,2],
                                                                                list_modalities=['A','B','C'],
                                                                                beta1=np.array([-8,7,6]),beta2=np.array([4,-7,3]),beta3=np.array([2,-1,2])
                                                                                            )
    X_final = np.hstack((Xf,feature_cat_uniform.reshape(-1,1)))
    X_final_num = np.hstack((Xf,feature_cat_uniform_numeric.reshape(-1,1)))
    feature_cat_uniform2, feature_cat_uniform_numeric2 = generate_synthetic_features_logreg_triple(X=Xf,index_informatives=[0,1,2],
                                                                                list_modalities=['D','E','F'],
                                                                                beta1=np.array([-4,5,6]),beta2=np.array([6,-3,2]),beta3=np.array([1,5,-1])
                                                                                            )
    X_final = np.hstack((X_final,feature_cat_uniform2.reshape(-1,1)))
    X_final_num = np.hstack((X_final_num,feature_cat_uniform_numeric2.reshape(-1,1)))
    
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(X_final[:,-2:])
    print(enc.get_feature_names_out())
    X_final_enc = np.hstack((Xf,X_final_cat_enc))
    
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives.extend([-(i+1) for i in range(n_modalities)])
    beta = [11,-8.1,-9,-1,8,5,-3,-5,2,6]
    print(list_index_informatives)
    print(beta[:(n_modalities+3)])
    print(X_final_enc[:,list_index_informatives].shape)

    probas = generate_synthetic_features_logreg_2(X=X_final_enc,index_informatives=list_index_informatives,
                                                  list_modalities=['No','Yes'],beta=beta[:(n_modalities+3)],intercept=-5)
    #y_num = np.random.normal(loc=0.5,scale=0.25,size=n_samples)
    rf = RandomForestRegressor(random_state=1234,max_depth=None)
    rf.fit(X_final_enc[:,list_index_informatives],probas)
    print("Mean depth : ", np.mean([rf.estimators_[i].get_depth() for i in range(len(rf.estimators_))]) )
    return rf,enc


def generate_initial_data_twocat(dimension,n_samples,rf,enc,random_state=24,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension-3):
        np.random.seed(seed=random_state+i+1)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
        
    feature_cat_uniform, feature_cat_uniform_numeric = generate_synthetic_features_logreg_triple(X=Xf,index_informatives=[0,1,2],
                                                                                list_modalities=['A','B','C'],
                                                                                beta1=np.array([-8,7,6]),beta2=np.array([4,-7,3]),beta3=np.array([2,-1,2])
                                                                                            )
    X_final = np.hstack((Xf,feature_cat_uniform.reshape(-1,1)))
    X_final_num = np.hstack((Xf,feature_cat_uniform_numeric.reshape(-1,1)))
    feature_cat_uniform2, feature_cat_uniform_numeric2 = generate_synthetic_features_logreg_triple(X=Xf,index_informatives=[0,1,2],
                                                                                list_modalities=['D','E','F'],
                                                                                beta1=np.array([-4,5,6]),beta2=np.array([6,-3,2]),beta3=np.array([1,5,-1])
                                                                                            )
    X_final = np.hstack((X_final,feature_cat_uniform2.reshape(-1,1)))
    X_final_num = np.hstack((X_final_num,feature_cat_uniform_numeric2.reshape(-1,1)))

    X_final_cat_enc = enc.transform(X_final[:,-2:])
    X_final_enc = np.hstack((Xf,X_final_cat_enc))
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives.extend([-(i+1) for i in range(n_modalities)])
    
    target_proba = rf.predict(X_final_enc[:,list_index_informatives])
    target = proba_to_label(y_pred_probas=target_proba, treshold=0.8)
    if verbose==1:
        print('Composition of the target ', Counter(target))
        print('Composition of categorical feature 1 : ', Counter(feature_cat_uniform))
        print('Composition of categorical feature 2 : ', Counter(feature_cat_uniform2))
        print('###########')
        print('Composition of the target ', Counter(target))
        print('Composition of categorical feature 1 : ', Counter(feature_cat_uniform))
        print('Composition of categorical feature 2 : ', Counter(feature_cat_uniform2))
        print('***************')

    X_final,target = subsample_to_ratio(X=X_final, y=target, ratio=0.1, seed_sub=random_state)

    if verbose==2:
        print('Composition of the target ', Counter(target))
        print('Composition of categorical feature 1 : ', Counter(feature_cat_uniform))
        print('Composition of categorical feature 2 : ', Counter(feature_cat_uniform2))
        print('###########')
        
    return X_final,target, target


def generate_synthetic_features_multinomial_quadruple(X,index_informatives,list_modalities=['A','B','C','D'],
                                              beta1=np.array([-8,7,6]),beta2=np.array([8,-7,6]),beta3=np.array([8,7,-6]),beta4=np.array([-3,-2,8]),
                                                   intercept=-2):

        
    linear1 = my_log_reg(X[:,index_informatives],beta=beta1,intercept=intercept)
    linear2 = my_log_reg(X[:,index_informatives],beta=beta2,intercept=intercept)
    linear3 = my_log_reg(X[:,index_informatives],beta=beta3,intercept=intercept)
    linear4 = my_log_reg(X[:,index_informatives],beta=beta4,intercept=intercept)
    sum_linear = linear1 + linear2 + linear3
    probas1 = linear1 / (1 + sum_linear)
    probas2 = linear2 / (1 + sum_linear)
    probas3 = linear3 / (1 + sum_linear)
    probas4 = 1 / (1 + sum_linear)
    #print('Somme des 4 probas : ', probas1+probas2+probas3+probas4)
    array_probas = np.array((probas1,probas2,probas3,probas4)).T
    #print('array_probas shape :', array_probas.shape)

    pred_log = np.array([np.random.multinomial(n=1,pvals=array_probas[i,:]) for i in range(len(array_probas))])
    #pred_log = np.random.multinomial(n=1,pvals=array_probas)
    #pred_log = np.apply_along_axis(np.random.multinomial,1,array_probas,{'n':1})
    #print('pred_log shape : ',pred_log.shape)
    #print('pred_log',pred_log)
    
    array_argmax = np.argmax(pred_log,axis=1)
    #print('array_argmax shape : ',array_argmax.shape)
    #print('array_argmax',array_argmax)

    array_final = np.char.replace(array_argmax.astype(str), '0',list_modalities[0])
    array_final = np.char.replace(array_final.astype(str), '1',list_modalities[1])
    array_final = np.char.replace(array_final.astype(str), '2',list_modalities[2])
    array_final = np.char.replace(array_final.astype(str), '3',list_modalities[3])
    #print('array_final shape ',array_final)


def generate_synthetic_features_multinomial_nonuple(X,index_informatives,list_modalities=['Ae','Bd','Af','Ce','Ad','Be','Bf','Cd','Cf'],
                                                      beta1=np.array([-8,7,6]),beta2=np.array([8,-7,6]),beta3=np.array([8,7,-6]),
                                                      beta4=np.array([-3,-2,8]),beta5=np.array([2,5,1]),beta6=np.array([3,2,8]),
                                                      beta7=np.array([7,6,6]),beta8=np.array([1,3,1]),beta9=np.array([1,1,9]),
                                                      intercept=-2):

        
    linear1 = my_log_reg(X[:,index_informatives],beta=beta1,intercept=intercept)
    linear2 = my_log_reg(X[:,index_informatives],beta=beta2,intercept=intercept)
    linear3 = my_log_reg(X[:,index_informatives],beta=beta3,intercept=intercept)
    linear4 = my_log_reg(X[:,index_informatives],beta=beta4,intercept=intercept)
    linear5 = my_log_reg(X[:,index_informatives],beta=beta5,intercept=intercept)
    linear6 = my_log_reg(X[:,index_informatives],beta=beta6,intercept=intercept)
    linear7 = my_log_reg(X[:,index_informatives],beta=beta7,intercept=intercept)
    linear8 = my_log_reg(X[:,index_informatives],beta=beta8,intercept=intercept)
    sum_linear = linear1 + linear2 + linear3 + linear4 + linear5 + linear6 + linear7 + linear8
    probas1 = linear1 / (1 + sum_linear)
    probas2 = linear2 / (1 + sum_linear)
    probas3 = linear3 / (1 + sum_linear)
    probas4 = linear4 / (1 + sum_linear)
    probas5 = linear5 / (1 + sum_linear)
    probas6 = linear6 / (1 + sum_linear)
    probas7 = linear7 / (1 + sum_linear)
    probas8 = linear8 / (1 + sum_linear)
    probas9 = 1 / (1 + sum_linear)
    #print('Somme des 4 probas : ', probas1+probas2+probas3+probas4)
    array_probas = np.array((probas1,probas2,probas3,probas4,probas5,probas6,probas7,probas8,probas9)).T
    #print('array_probas shape :', array_probas.shape)

    pred_log = np.array([np.random.multinomial(n=1,pvals=array_probas[i,:]) for i in range(len(array_probas))])
    #pred_log = np.random.multinomial(n=1,pvals=array_probas)
    #pred_log = np.apply_along_axis(np.random.multinomial,1,array_probas,{'n':1})
    #print('pred_log shape : ',pred_log.shape)
    #print('pred_log',pred_log)
    
    array_argmax = np.argmax(pred_log,axis=1)
    #print('array_argmax shape : ',array_argmax.shape)
    #print('array_argmax',array_argmax)

    array_final = np.char.replace(array_argmax.astype(str), '0',list_modalities[0])
    array_final = np.char.replace(array_final.astype(str), '1',list_modalities[1])
    array_final = np.char.replace(array_final.astype(str), '2',list_modalities[2])
    array_final = np.char.replace(array_final.astype(str), '3',list_modalities[3])
    array_final = np.char.replace(array_final.astype(str), '4',list_modalities[4])
    array_final = np.char.replace(array_final.astype(str), '5',list_modalities[5])
    array_final = np.char.replace(array_final.astype(str), '6',list_modalities[6])
    array_final = np.char.replace(array_final.astype(str), '7',list_modalities[7])
    array_final = np.char.replace(array_final.astype(str), '8',list_modalities[8])
    #print('array_final shape ',array_final)
    
    
    return array_final,array_argmax

def generate_initial_data_twocat_lgbm5_old(dimension,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension-3):
        np.random.seed(seed=random_state+i+1)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
    ### Feature categorical
        
    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial_nonuple(
        X=Xf,index_informatives=[0,1,2],list_modalities=['Ae','Bd','Af','Ce','Ad','Be','Bf','Cd','Cf'],beta1=np.array([1,3,2]),
        beta2=np.array([4,-7,3]),beta3=np.array([5,-1,6]),beta4=np.array([3,2,1]),beta5=np.array([2,5,1]),
        beta6=np.array([3,2,8]),beta7=np.array([7,6,6]),beta8=np.array([1,3,1]),beta9=np.array([1,1,9]),intercept=-2
    )
    #print('z_feature_cat_uniform shape :',z_feature_cat_uniform_numeric.shape)
    
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    print(enc.get_feature_names_out())
    X_final_enc = np.hstack((Xf,X_final_cat_enc))
    
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    print(list_index_informatives_cat)
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    beta = [3,5.1,4,-100,4,3,3.2,-100,-100,-100,5,-100]
    #print('list_index_informatives : ',list_index_informatives)
    #print(beta[:(n_modalities+3)])
    #print(X_final_enc[:,list_index_informatives].shape)

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=0.5,intercept=-11.6 # intercept=-3
                                                               )
    #print('target_numeric.shape',target_numeric.shape)
    #print('target_numeric', target_numeric)
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    return X_final,target,target_numeric


def generate_initial_data_twocat_lgbm5_oldcoeffes(dimension,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension-7):
        np.random.seed(seed=random_state+i+1)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
    ### Feature categorical
        
    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial_nonuple(
        X=Xf,index_informatives=[0,1,2],list_modalities=['Ae','Bd','Af','Ce','Ad','Be','Bf','Cd','Cf'],beta1=np.array([1,3,2]),
        beta2=np.array([4,-7,3]),beta3=np.array([5,-1,6]),beta4=np.array([3,2,1]),beta5=np.array([2,5,1]),
        beta6=np.array([3,2,8]),beta7=np.array([7,6,6]),beta8=np.array([1,3,1]),beta9=np.array([1,1,9]),intercept=-2
    )
    #print('z_feature_cat_uniform shape :',z_feature_cat_uniform_numeric.shape)
    
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    print(enc.get_feature_names_out())
    X_final_enc = np.hstack((Xf,X_final_cat_enc))
    
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    print('list_index_informatives : ',list_index_informatives)
    beta = [3,-5.1,-4,-8,4,-3,3.2,-8,-7,-9,5,-10]
    #print('list_index_informatives : ',list_index_informatives)
    #print(beta[:(n_modalities+3)])
    #print(X_final_enc[:,list_index_informatives].shape)

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=0.5,intercept=-1.9 # intercept=-3
                                                               )
    #print('target_numeric.shape',target_numeric.shape)
    #print('target_numeric', target_numeric)
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    #return X_final,target,target_numeric
    return X_final,target,target_numeric

def generate_initial_data_twocat_lgbm5_19_11_2024(dimension,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension-7):
        np.random.seed(seed=random_state+i+1)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
    ### Feature categorical
        
    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial_nonuple(
        X=Xf,index_informatives=[0,1,2],list_modalities=['Ae','Bd','Af','Ce','Ad','Be','Bf','Cd','Cf'],beta1=np.array([1,3,2]),
        beta2=np.array([4,-7,3]),beta3=np.array([5,-1,6]),beta4=np.array([3,2,1]),beta5=np.array([2,5,1]),
        beta6=np.array([3,2,8]),beta7=np.array([7,6,6]),beta8=np.array([1,3,1]),beta9=np.array([1,1,9]),intercept=-2
    )
    #print('z_feature_cat_uniform shape :',z_feature_cat_uniform_numeric.shape)
    
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    print(enc.get_feature_names_out())
    X_final_enc = np.hstack((Xf,X_final_cat_enc))
    
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    print('list_index_informatives : ',list_index_informatives)
    beta = [3,-5.1,-4,-8,-1.2,1.3,2.4,-8,-8,-9,-0.5,-10]
    #print('list_index_informatives : ',list_index_informatives)
    #print(beta[:(n_modalities+3)])
    #print(X_final_enc[:,list_index_informatives].shape)

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=0.5,intercept=0.9 # intercept=-3
                                                               )
    #print('target_numeric.shape',target_numeric.shape)
    #print('target_numeric', target_numeric)
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    #return X_final,target,target_numeric
    return X_final,target,target_numeric


def generate_initial_data_twocat_lgbm5(dimension_continuous,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension_continuous-1):
        np.random.seed(seed=random_state+i+1)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
    ### Feature categorical
        
    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial_nonuple(
        X=Xf,index_informatives=[0,1,2],list_modalities=['Ae','Bd','Af','Ce','Ad','Be','Bf','Cd','Cf'],beta1=np.array([4.1,-1.1,10.7]),
        beta2=np.array([4.3,-2.1,11.1]),beta3=np.array([4.8,-0.7,9.7]),beta4=np.array([5,-1,10]),beta5=np.array([-5,-1,10]),
        beta6=np.array([-5,-1,10]),beta7=np.array([4.3,-2.1,11.1]),beta8=np.array([4.3,-2.1,11.1]),beta9=np.array([1,1,9]),intercept=2
    )
    #print('z_feature_cat_uniform shape :',z_feature_cat_uniform_numeric.shape)
    
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    print(enc.get_feature_names_out())
    X_final_enc = np.hstack((Xf,X_final_cat_enc))
    
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    print('list_index_informatives : ',list_index_informatives)
    beta = [8,-10.1,-9,-11.3,-10.2,1.3,1.8,-11.8,-12.2,-10.7,0.9,-12]
    #print('list_index_informatives : ',list_index_informatives)
    #print(beta[:(n_modalities+3)])
    #print(X_final_enc[:,list_index_informatives].shape)

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=0.5,intercept=0.5 # intercept=-3
                                                               )
    
    #print('target_numeric.shape',target_numeric.shape)
    #print('target_numeric', target_numeric)
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    #return X_final,target,target_numeric
    return X_final,target,target_numeric