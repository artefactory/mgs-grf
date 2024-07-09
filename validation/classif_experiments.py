import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import shuffle
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.metrics import (roc_curve,auc,precision_recall_curve,roc_auc_score)
from sklearn.metrics import PrecisionRecallDisplay,RocCurveDisplay
from scipy import interpolate


def proba_to_label(y_pred_probas, treshold=0.5):  # apply_threshold ?
    """_summary_

    Parameters
    ----------
    y_pred_probas : _type_
        _description_
    treshold : float, optional
        _description_, by default 0.5

    Returns
    -------
    _type_
        _description_
    """
    # Personnally I would do it in NumPy:
    return np.array(np.array(y_pred_probas) >= treshold, dtype=int)


def subsample_to_ratio_indices(
    X,
    y,
    ratio,
    seed_sub,
    output_dir_subsampling,
    name_subsampling_file,
    has_previous_under_sampling=False,
    previous_under_sampling=None,
):
    """docstring: lots of arguments, important to be clear !

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    ratio : _type_
        _description_
    seed_sub : _type_
        _description_
    output_dir_subsampling : _type_
        _description_
    name_subsampling_file : _type_
        _description_
    has_previous_under_sampling : bool, optional
        _description_, by default False
    previous_under_sampling : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    Path(output_dir_subsampling).mkdir(
        parents=True, exist_ok=True
    )  ## build the directory if it does not exists
    if has_previous_under_sampling:
        X_under, y_under = X[previous_under_sampling, :], y[previous_under_sampling]
        X_negatifs = X_under[np.array(1 - y_under, dtype=bool)]  
    else:
        X_negatifs = X[np.array(1 - y, dtype=bool)]

    np.random.seed(seed=seed_sub)
    n_undersampling_sub = int(
        (ratio * len(X_negatifs)) / (1 - ratio)
    )  ## compute number of sample to keeep
    ##int() inr order to have upper integer part
    df_X = pd.DataFrame(data=X)
    if previous_under_sampling is not None:
        indices_positifs_kept = np.random.choice(
            df_X.loc[previous_under_sampling]
            .loc[np.array(y_under, dtype=bool)]
            .index.values,
            size=n_undersampling_sub,
            replace=False,
        )
        indices_negatifs_kept = (
            df_X.loc[previous_under_sampling]
            .loc[np.array(1 - y_under, dtype=bool), :]
            .index.values
        )
        indices_kept = np.hstack((indices_positifs_kept, indices_negatifs_kept))
    else:
        indices_positifs_kept = np.random.choice(
            df_X.loc[np.array(y, dtype=bool)].index.to_numpy(),
            size=n_undersampling_sub,
            replace=False,
        )
        indices_negatifs_kept = df_X.loc[np.array(1 - y, dtype=bool)].index.to_numpy()
        indices_kept = np.hstack(
            (indices_positifs_kept, indices_negatifs_kept)
        )  # kept_indexes

    # set a default location + name for cache ?
    np.save(
        os.path.join(output_dir_subsampling, name_subsampling_file + ".npy"),
        indices_kept,
    )
    return indices_kept


def read_subsampling_indices(
    X, y, dir_subsampling, name_subsampling_file, get_indexes=False
):
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    dir_subsampling : _type_
        _description_
    name_subsampling_file : _type_
        _description_
    get_indexes : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    indexes_subsampling = np.load(
        os.path.join(dir_subsampling, name_subsampling_file + ".npy")
    )
    if get_indexes:
        return indexes_subsampling, X[indexes_subsampling, :], y[indexes_subsampling]
    else:
        return X[indexes_subsampling, :], y[indexes_subsampling]


####### run_eval ##########
def subsample_to_ratio(X, y, ratio, seed_sub):
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    ratio : _type_
        _description_
    seed_sub : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
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


def run_eval(
    output_dir,
    name_file,
    X,
    y,
    list_oversampling_and_params,
    splitter,
    #subsample_ratios=[0.2, 0.1, 0.01],
    #subsample_seeds=[11, 9, 5],
    to_standard_scale=True,
    categorical_features=None
):
    """
    Main function of the procol.
    output_dir is the path where the output files will be stored.
    list_oversampling_and_params is a list composed of tuple like (name, function, function_params, classifier).
    to_standard_scale is a boolean.

    Paramters
    ---------

    Returns
    -------
    """

    ################## INITIALISATION #################
    n_strategy = len(list_oversampling_and_params)
    list_names_oversamplings = ["y_true"] + [
        config[0] for config in list_oversampling_and_params
    ]
    list_names_oversamplings.append("fold")

    list_all_preds = [[] for i in range(n_strategy + 2)]
    list_tree_depth = []
    list_tree_depth_name = []

    X_copy, y_copy = X.copy(), y.copy()

    folds = list(splitter.split(X_copy, y_copy))
    ##############################################
    ######## Start protocol by strategy    #######
    ##############################################
    for i, (
        oversampling_name,
        oversampling_func,
        oversampling_params,
        model,
    ) in enumerate(list_oversampling_and_params):
        for fold, (train, test) in enumerate(folds):
            ################## Folds data are prepared #############
            X_train, y_train = X_copy[train], y_copy[train]
            X_test = X_copy[test]
            if to_standard_scale:
                scaler = StandardScaler()
                if categorical_features is None:
                    X_train = scaler.fit_transform(X_train)
                else:
                    bool_mask = np.ones((X_train.shape[1]), dtype=bool)
                    bool_mask[categorical_features]= False
                    X_train[:,bool_mask] = scaler.fit_transform(X_train[:,bool_mask]) ## continuous features only

            X_res, y_res = oversampling_func.fit_resample(
                X=X_train, y=y_train, **oversampling_params
            )
            ######### Run of the given fold ###############

            # Is shuffling useful within a fold isn't integrated in RF model ?
            X_res, y_res = shuffle(X_res, y_res,random_state=0)  # to put in oversampling_func
            model.fit(X_res, y_res)
            forest = hasattr(model, "estimators_") and hasattr(
                model.estimators_[0], "get_depth"
            )
            if forest:
                curent_tree_depth = [
                    estimator.get_depth() for estimator in model.estimators_
                ]
                list_tree_depth.append(curent_tree_depth)
                list_tree_depth_name.append(oversampling_name)

            if to_standard_scale:
                if categorical_features is None:
                    X_test = scaler.transform(X_test)
                else:
                    bool_mask = np.ones((X_test.shape[1]), dtype=bool)
                    bool_mask[categorical_features]= False
                    X_test[:,bool_mask] = scaler.transform(X_test[:,bool_mask]) ## continuous features only
            y_pred_probas = model.predict_proba(X_test)[:, 1]

            ######## Results are saved ###################
            list_all_preds[i + 1].extend(y_pred_probas)
            if i == 0:
                list_all_preds[-1].extend(
                    np.full((len(test),), fold)
                )  # save information of the ciurrent testing fold
                list_all_preds[0].extend(
                    y_copy[test]
                )  # save the information of the target value
    if len(list_tree_depth) != 0:
        pd.DataFrame(np.array(list_tree_depth).T, columns=list_tree_depth_name).to_csv(
            os.path.join(output_dir, "depth" + name_file[:-4] + ".csv")
        )
    runs_path_file_strats = os.path.join(output_dir, "preds_" + name_file)
    np.save(runs_path_file_strats, np.array(list_all_preds).T)
    np.save(
        os.path.join(output_dir, "name_strats" + name_file), list_names_oversamplings
    )


def compute_metrics(output_dir, name_file, list_metric):
    """_summary_

    Parameters
    ----------
    output_dir : _type_
        _description_
    name_file : _type_
        _description_
    list_metric : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    n_metric = len(list_metric)
    metrics_names = []
    for m in range(n_metric):
        metrics_names.append(list_metric[m][1])
    oversample_strategies = np.load(os.path.join(output_dir, "name_strats" + name_file))
    predictions_by_strategy = np.load(os.path.join(output_dir, "preds_" + name_file))
    df_all = pd.DataFrame(predictions_by_strategy, columns=oversample_strategies)

    name_col_strategies = df_all.columns.tolist()
    name_col_strategies.remove("y_true")
    name_col_strategies.remove("fold")
    # We remove  'y_true' and 'fold'

    array_resultats_metrics = np.zeros((n_metric, len(name_col_strategies)))
    array_resultats_metrics_std = np.zeros((n_metric, len(name_col_strategies)))
    for k in range(n_metric):
        for col_number, col_name in enumerate(name_col_strategies):
            ### Mean of the metrics on the 5 test folds:
            list_value = []
            for j in range(5):
                df = df_all[df_all["fold"] == j]
                y_true = df["y_true"].tolist()
                pred_probas_all = df[col_name].tolist()
                y_pred = proba_to_label(y_pred_probas=pred_probas_all, treshold=0.5)

                if list_metric[k][2] == "pred":
                    value_metric = list_metric[k][0](y_true=y_true, y_pred=y_pred)
                else:
                    value_metric = list_metric[k][0](
                        y_true=y_true, y_score=pred_probas_all
                    )
                list_value.append(value_metric)
            array_resultats_metrics[k, col_number] = np.mean(list_value)
            array_resultats_metrics_std[k, col_number] = np.std(list_value)

    df_mean_metric = pd.DataFrame(
        array_resultats_metrics, columns=name_col_strategies, index=metrics_names
    )
    df_std_metric = pd.DataFrame(
        array_resultats_metrics_std,
        columns=name_col_strategies,
        index=metrics_names,
    )
    return df_mean_metric, df_std_metric


def compute_metrics_several_protocols(
    output_dir, init_name_file, list_metric, bool_roc_auc_only=True, n_iter=100
):
    """_summary_

    Parameters
    ----------
    output_dir : _type_
        _description_
    init_name_file : _type_
        _description_
    list_metric : _type_
        _description_
    bool_roc_auc_only : bool, optional
        _description_, by default True
    n_iter : int, optional
        _description_, by default 100

    Returns
    -------
    _type_
        _description_
    """
    list_res = []

    ######### CASE  ROC AUC only is computed ######
    if bool_roc_auc_only is True:
        for i in range(n_iter):
            name_file = init_name_file + str(i) + ".npy"
            df_metrics_mean, df_metrics_std = compute_metrics(
                output_dir=output_dir,
                name_file=name_file,
                list_metric=[(roc_auc_score, "roc_auc", "proba")],
            )
            list_res.append(df_metrics_mean.to_numpy())

        name_cols = df_metrics_mean.columns
        array_res = np.array(list_res)
        df_final_mean = pd.DataFrame(
            np.mean(array_res, axis=0).reshape((1, -1)), columns=name_cols
        )
        df_final_std = pd.DataFrame(
            np.std(array_res, axis=0).reshape((1, -1)), columns=name_cols
        )
        df_final_mean.index = ["ROC AUC"]
        df_final_std.index = ["ROC AUC"]

    ######## CASE all the metrics are computed #######
    else:
        for i in range(n_iter):
            name_file = init_name_file + str(i) + ".npy"
            df_metrics_mean, df_metrics_std = compute_metrics(
                output_dir=output_dir, name_file=name_file, list_metric=list_metric
            )

            list_res.append(df_metrics_mean.to_numpy())

        name_cols = df_metrics_mean.columns
        array_res = np.array(list_res)
        df_final_mean = pd.DataFrame(np.mean(array_res, axis=0), columns=name_cols)
        df_final_std = pd.DataFrame(np.std(array_res, axis=0), columns=name_cols)
        list_metric_func, list_metric_name, list_metric_type = zip(*list_metric)
        df_final_mean.index = list_metric_name
        df_final_std.index = list_metric_name

    return df_final_mean, df_final_std


class PaperTimeSeriesSplit(TimeSeriesSplit):
    """
    The starting split can be chosen with this child class from TimeSeriesSplit.
    """

    def __init__(
        self, n_splits=10, starting_split=5, max_train_size=None, test_size=None, gap=0
    ):
        """ """
        super().__init__(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
            gap=gap,
        )
        self.starting_split = starting_split

    def split(self, X, y=None, groups=None):
        """ """
        folds = list(super().split(X))
        folds_from_starting_split = folds[self.starting_split :]
        return folds_from_starting_split


class PaperTimeSeriesSplitWithGroupOut(TimeSeriesSplit):
    """
    MyTimeSeriesSplit with group out on col_name_id.
    All the samples with ID that have been seen during the training phase, are removed of the test set.
    """

    def __init__(
        self,
        meta_df,
        col_name_id,
        n_splits=10,
        starting_split=5,
        max_train_size=None,
        test_size=None,
        gap=0,
    ):
        """
        col_name_id : name of the column containing the ID
        """
        super().__init__(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
            gap=gap,
        )
        self.starting_split = starting_split
        self.meta_df = meta_df
        self.col_name_id = col_name_id

    def split(self, X, y=None, groups=None):
        """ """
        folds = list(super().split(X))
        folds_from_starting_split = folds[self.starting_split :]
        final_folds_from_starting_split = []
        for fold, (train_index, test_index) in enumerate(folds_from_starting_split):
            # Split:
            meta_df_train, meta_df_test = (
                self.meta_df.iloc[train_index],
                self.meta_df.iloc[test_index],
            )
            # Samples with ID that have been seen during training are removed from the test set:
            id_in_train = meta_df_train[self.col_name_id].unique().tolist()
            test_indices_to_keep = meta_df_test.index[
                ~meta_df_test[self.col_name_id].isin(id_in_train)
            ].tolist()
            tmp = (train_index, test_indices_to_keep)
            final_folds_from_starting_split.append(tmp)

        return final_folds_from_starting_split


def depth_func_linspace(min_value,max_value,size=10,add_border=False):
    list_depth = np.linspace(min_value,max_value,size, dtype=int).tolist()
    if add_border:
        border_array = [max_value-3,max_value-2,max_value-1,max_value,None]
    else :
        border_array=[None]
    list_depth.extend(border_array)
    return list(dict.fromkeys(list_depth))

def plot_curves(output_dir,start_filename,n_iter,stategies_to_show=None,names_stategies_to_show=None,show_pr=False,show_auc_curves=True,to_show=True,value_alpha=0.2,kind_interpolation = 'linear'):
    """_summary_

    Parameters
    ----------
    output_dir : str
        path direcory
    name_file : str
        standard names of the .npy files inside output_dir
    n_iter :  int
        number of file to be read by the function
    stategies_to_show : list of str or None (default value)
        When set to None, show all the strategies seen in each file. When set to a list of str, read only the specified startegies
    show_pr : bool
        Show PR curves by default and ROC curves otherwise
    """
    filename_0 = start_filename + str(0) +'.npy'
    if stategies_to_show is None :
        stategies_to_show = np.load(
            os.path.join(output_dir, "name_strats" + filename_0)
        ).tolist()
        stategies_to_show.remove('fold') # remove fold column which is not a strategy
        stategies_to_show.remove('y_true') # remove y_true column which is not a strategy
    if names_stategies_to_show is None :
        names_stategies_to_show = stategies_to_show
    
    list_names_oversamplings = np.load(
        os.path.join(output_dir, "name_strats" + filename_0)
    )
        
    list_fpr = np.arange(start=0,stop=1.01,step=0.01)
    list_recall = np.arange(start=0,stop=1.01,step=0.01)
    array_interpolated_quantity = np.zeros((n_iter,len(list_recall),len(stategies_to_show)))
    array_quantity_auc = np.zeros((n_iter,len(stategies_to_show)))
    for i in range(n_iter):
        filename = start_filename + str(i) +'.npy'
        array_all_preds_strats_final = np.load(
            os.path.join(output_dir, "preds_" + filename)
        )
        df_all = pd.DataFrame(
            array_all_preds_strats_final, columns=list_names_oversamplings
        )
            
        for j,col in enumerate(stategies_to_show):
            array_interpolated_quantity_folds = np.zeros((5,len(list_recall)))
            list_auc_folds = []
            for fold in range(5):
                df = df_all[df_all["fold"] == fold]
                y_true = df["y_true"].tolist()
                pred_probas_col = df[col].tolist()
                
                if show_pr: ## PR Curves case
                    prec, rec, tresh = precision_recall_curve(y_true, pred_probas_col)
                    pr_auc = auc(rec, prec)
                    interpolation_func = interpolate.interp1d(np.flip(rec), np.flip(prec) ,kind=kind_interpolation)
                    prec_interpolated = interpolation_func(list_recall)
                    #array_interpolated_quantity_folds[fold,:] = prec_interpolated
                    array_interpolated_quantity_folds[fold,:] = np.flip(prec_interpolated)
                    list_auc_folds.append(pr_auc)
                else :## ROC Curves case
                    fpr, tpr, _ = roc_curve(y_true, pred_probas_col)
                    interpolation_func = interpolate.interp1d(fpr, tpr,kind=kind_interpolation)
                    tpr_interpolated = interpolation_func(list_fpr)  
                    array_interpolated_quantity_folds[fold,:] = tpr_interpolated
                    roc_auc = roc_auc_score(y_true, pred_probas_col)
                    list_auc_folds.append(roc_auc)
                    
            array_interpolated_quantity[i,:,j]=array_interpolated_quantity_folds.mean(axis=0) ## the mean interpolated over the 5 fold are averaged
            array_quantity_auc[i,j] = np.mean(list_auc_folds)
    mean_final_prec = array_interpolated_quantity.mean(axis=0) ## interpolated precisions over the n_iter ietartions are averaged by strategy
    std_final_prec = array_interpolated_quantity.std(axis=0)
    ########### Plotting curves ##############
    if to_show:
        plt.figure(figsize=(10,6))
    for h,col in enumerate(names_stategies_to_show):
        if show_pr: ## PR Curves case
            if show_auc_curves :
                pr_auc_col = auc(np.flip(list_recall),mean_final_prec[:,h])
            else :
                pr_auc_col = array_quantity_auc[:,h].mean()
            lab_col = col + ' AUC='+ str(round(pr_auc_col,3))
            #disp = PrecisionRecallDisplay(precision=mean_final_prec[:,h], recall=np.flip(list_recall))
            #disp.plot()
            plt.plot(np.flip(list_recall), mean_final_prec[:,h], label=lab_col)
            plt.fill_between(np.flip(list_recall), mean_final_prec[:,h] + std_final_prec[:,h],
                             mean_final_prec[:,h] - std_final_prec[:,h], alpha=value_alpha,step='pre') #color='grey'
        else: ## ROC Curves case
            if show_auc_curves :
                pr_auc_col = auc(list_fpr,mean_final_prec[:,h])
            else :
                pr_auc_col = array_quantity_auc[:,h].mean()
            lab_col = col + ' AUC='+ str(round(pr_auc_col,3))
            plt.scatter(list_fpr,mean_final_prec[:,h],label=lab_col)
            plt.fill_between(list_fpr, mean_final_prec[:,h] + std_final_prec[:,h],
                             mean_final_prec[:,h] - std_final_prec[:,h], alpha=value_alpha,step='pre') #color='grey'
    #################### Add legend or not (for tuned function ploting) ##################
    if to_show:
        if show_pr:
            plt.legend(loc='best', fontsize='small')
            plt.title("PR Curves", weight="bold", fontsize=15)
            plt.xlabel("Recall", fontsize=12)
            plt.ylabel("Precision", fontsize=12)
        else:
            plt.legend(loc='best', fontsize='small')
            plt.title("ROC Curves", weight="bold", fontsize=15)
            plt.xlabel("False Positive Rate (FPR)", fontsize=12)
            plt.ylabel("True Positive Rate (TPR)", fontsize=12)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.show()


def plot_curves_tuned(output_dir,start_filename,n_iter,list_name_strat,list_name_strat_inside_file,list_name_strat_to_show=None,show_pr=False,
                      show_auc_curves=True,value_alpha=0.2,kind_interpolation='linear'):
    plt.figure(figsize=(10,6))
    if list_name_strat_to_show is None:
        list_name_strat_to_show = list_name_strat_inside_file
    for i,strat in enumerate(list_name_strat):
        curr_start_output_dir=os.path.join(output_dir,strat,'RF_100')
        plot_curves(output_dir=curr_start_output_dir,start_filename=start_filename,n_iter=n_iter,
                    stategies_to_show=[list_name_strat_inside_file[i]],names_stategies_to_show=[list_name_strat_to_show[i]],show_pr=show_pr,show_auc_curves=show_auc_curves,
                    to_show=False,value_alpha=value_alpha,kind_interpolation=kind_interpolation)
    
    if show_pr:
        plt.legend(loc='best', fontsize='small')
        plt.title("PR Curves", weight="bold", fontsize=15)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
    else:
        plt.legend(loc='best', fontsize='small')
        plt.title("ROC Curves", weight="bold", fontsize=15)
        plt.xlabel("False Positive Rate (FPR)", fontsize=12)
        plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.show()


from sklearn.metrics import (precision_recall_curve,auc)
def pr_auc_custom(y_true, y_score):
    precision, recall, tresh = precision_recall_curve(y_true, y_score)
    res_auc = auc(recall,precision)
    return res_auc