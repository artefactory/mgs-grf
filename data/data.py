import os

import pandas as pd
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
        df_diabete = pd.read_csv(
            os.path.join(DATA_DIR, filename)
        )
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
        names=['Sequence_Name','mcg','gvh','alm','mit','erl','pox','vac','nuc','localization_site'],
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
    df_magic.replace({'class':{b'h':0,b'g':1}},inplace=True)
    X_magic = df_magic.drop(['class'],axis=1).to_numpy()   
    y_magic = df_magic[['class']].to_numpy().ravel()

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
    df_cal_housing.replace({'price':{b'True':1,b'False':0}},inplace=True)
    X_cal_housing = df_cal_housing.drop(['price'],axis=1).to_numpy()
    y_cal_housing = df_cal_housing[['price']].to_numpy().ravel()

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
    df_house_16H.replace({'binaryClass':{b'P':0,b'N':1}},inplace=True) 
    X_house_16H = df_house_16H.drop(['binaryClass'],axis=1).to_numpy()
    y_house_16H = df_house_16H[['binaryClass']].to_numpy().ravel()

    return X_house_16H, y_house_16H

def load_vehicle_data():
    """
    Load vehicle data set from data\dexternals folder
    All the file dowloaded/unziped from UCI should be in a folder named vehicle vehicle
    """
    names_vehicle = ['COMPACTNESS','CIRCULARITY','DISTANCE_CIRCULARITY','RADIUS_RATIO','PR_AXIS_ASPECT_RATIO','MAX_LENGTH_ASPECT_RATIO',
                     'SCATTER_RATIO','ELONGATEDNESS','PR_AXIS_RECTANGULARITY','MAX_LENGTH_RECTANGULARITY','SCALED_VARIANCE_ALONG_MAJOR_AXIS',
                     'SCALED_VARIANCE_ALONG_MINOR_AXIS','SCALED_RADIUS_OF_GYRATION','SKEWNESS_ABOUT_MAJOR_AXIS','SKEWNESS_ABOUT_MINOR_AXIS',
                     'KURTOSIS_ABOUT_MINOR_AXIS','KURTOSIS_ABOUT_MAJOR_AXIS','HOLLOWS_RATIO', 'Class']
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
    df_vehicle = pd.concat([df_tmp_1,df_tmp_2,df_tmp_3,df_tmp_4,df_tmp_5,df_tmp_6,df_tmp_7,df_tmp_8,df_tmp_9],axis=0)
    df_vehicle.columns = names_vehicle
    df_vehicle.replace({'Class':{'van':1,'saab':0,'bus':0,'opel':0}},inplace=True)
    X_vehicle = df_vehicle.drop(['Class'],axis=1).to_numpy()
    y_vehicle = df_vehicle[['Class']].to_numpy().ravel()

    return X_vehicle,y_vehicle

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
    df_b_cancer_w = pd.read_csv('/home/abdoulaye_sakho/Code/data/breast-cancer-wisconsin.data',sep=",",header=None)
    df_b_cancer_w.columns = ['Sample_code_number','Clump_thickness','Uniformity_of_cell_size',
                            'Uniformity_of_cell_shape','Marginal_adhesion','Single_epithelial_cell_size',
                            'Bare_nuclei','Bland_chromatin','Normal_nucleoli','Mitoses','Class']
    df_b_cancer_w.replace({'Class':{2:0,4:1}},inplace=True)
    df_b_cancer_w = df_b_cancer_w[df_b_cancer_w.Bare_nuclei != '?']
    df_b_cancer_w = df_b_cancer_w.astype({'Bare_nuclei': 'int64'})
    df_b_cancer_w.drop_duplicates(subset=["Sample_code_number"], keep='first',inplace=True)
    X_b_cancer_w =  df_b_cancer_w.drop(['Class'],axis=1).to_numpy()
    y_b_cancer_w =  df_b_cancer_w[['Class']].to_numpy().ravel()

    return X_b_cancer_w,y_b_cancer_w

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
    df_ionosphere.replace({34:{'g':0,'b':1}},inplace=True)
    X_ionosphere =  df_ionosphere.drop([1,34],axis=1).to_numpy()
    y_ionosphere =  df_ionosphere[[34]].to_numpy().ravel()

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
    meta_df_credit = df_credit[['Time']]
    X_credit = df_credit.drop(['Time','Class'],axis=1).to_numpy()
    y_credit = df_credit[['Class']].to_numpy().ravel()

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
    df_wine = pd.concat([X,y],axis=1)

    dict_mapping = {5:0, 6:0, 8:1}
    df_wine = df_wine[df_wine['quality'].isin([5,6,8])].copy()
    df_wine.replace({'quality':dict_mapping},inplace=True)
    X_wine = df_wine.drop(['quality'],axis=1).to_numpy()
    y_wine = df_wine[['quality']].to_numpy().ravel()
    return X_wine,y_wine


from ucimlrepo import fetch_ucirepo 
def load_BankMarketing_data():
    """
    Load BankChurners data set from UCI Irvine.
    """
    # fetch dataset 
    bank_marketing = fetch_ucirepo(id=222) 
    # data (as pandas dataframes) 
    X = bank_marketing.data.features 
    y = bank_marketing.data.targets 

    X.fillna('unknow',inplace=True) # fillna
    y.replace({'y':{'yes':1,'no':0}},inplace=True)
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
    
    X_bankChurners = df_bankchurners.drop(['CLIENTNUM','Attrition_Flag','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
                                         axis=1).to_numpy()
    y_bankChurners = df_bankchurners[["Attrition_Flag"]].replace({'Attrition_Flag':{'Existing Customer':0, 'Attrited Customer':1}}).to_numpy().ravel()
    return X_bankChurners, y_bankChurners



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
    df_telco_churn.replace({'TotalCharges':{' ':'0'}},inplace=True)
    df_telco_churn[["TotalCharges"]] = df_telco_churn[["TotalCharges"]].astype(float)
    
    X_telco = df_telco_churn.drop(['customerID','Churn'],axis=1).to_numpy()
    y_telco = df_telco_churn[["Churn"]].replace({'Churn':{'No':0, 'Yes':1}}).to_numpy().ravel()

    return X_telco,y_telco

