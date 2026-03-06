"""
This file contains the functions to compute the association metric and to plot it over dimensions.
The functions are specifically designed for being imported by the notebook experiments/protocols/notebook/res_asso.ipynb.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score,balanced_accuracy_score,average_precision_score
from sklearn.model_selection import LeaveOneOut
import lightgbm as lgb

from ..validation.classif_experiments import (compute_metrics_several_protocols,
                                            prec_at_recall_version3, prec_at_recall_version3_02)


list_metric = [
    #(precision_score,'precision','pred'),
    #(recall_score,'recall','pred'),
    #(balanced_accuracy_score,'balanced acc','pred'),
    (prec_at_recall_version3_02,'p@r=0.2','proba'),
    (prec_at_recall_version3,'p@r=0.5','proba'),
    (average_precision_score,'avg_pr_auc', 'proba'),
    (roc_auc_score,'roc_auc','proba') 
]

def compute_runs_over_dimensions(dimensions,strategies,init_output_dir_path,init_name_file,n_iter=20,n_fold=5):
    array_res_mean = np.zeros((len(dimensions),len(strategies)))
    array_res_std = np.zeros((len(dimensions),len(strategies)))
    
    for i,dimension in enumerate(dimensions) :
        output_dir_path =init_output_dir_path  +str(dimension)
        df_final_mean,df_final_std = compute_metrics_several_protocols(
            output_dir=output_dir_path,
            init_name_file=init_name_file,
            list_metric=list_metric,
            bool_roc_auc_only=False,n_iter=n_iter,n_fold=n_fold)
        for j,strat in enumerate(strategies) :
            array_res_mean[i,j] = df_final_mean.loc['avg_pr_auc'][strat]
            array_res_std[i,j] = df_final_std.loc['avg_pr_auc'][strat]
    
    array_res_mean_with_dim = np.hstack((np.array(dimensions).reshape(-1,1),array_res_mean))
    array_res_std_with_dim = np.hstack((np.array(dimensions).reshape(-1,1),array_res_std))
    columns_with_dim = ['Dimension']
    columns_with_dim.extend(strategies)
    df_res_mean= pd.DataFrame(array_res_mean_with_dim,columns=columns_with_dim)
    df_res_std= pd.DataFrame(array_res_std_with_dim,columns=columns_with_dim)
    return df_res_mean,df_res_std
    
def plot_(df,df_std,xlim=[-0.1,510],ylim=[0.55,1.0],fontsize=20,plot_error_fill=False,title='',
         name_strats_to_plot= None,to_save=False,name_file_saving='img.pdf'):
    plt.figure(figsize=(12,8))
    list_start = df.columns.tolist()[1:4]
    list_fmt= ['o','v','^','s','*','8']*((len(list_start) // 6)+1)
    list_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                   '#7f7f7f', '#bcbd22', '#17becf'] *((len(list_start) // 10)+1)
    for i,strat in enumerate(list_start):
        if plot_error_fill:
            if name_strats_to_plot :
                name_strat= name_strats_to_plot[i]
            else:
                name_strat=strat
            #plt.errorbar(df[['Dimension']].values.ravel(),df[[strat]].values.ravel(),
            #                 yerr=df_std[[strat]].values.ravel(),fmt=list_fmt[i], markersize=3,elinewidth=0.5, capsize=6,label=name_strat
            #                )
            plt.plot(df[['Dimension']].values.ravel(),df[[strat]].values.ravel(), 
                     marker=list_fmt[i],linestyle="--", label=name_strat,c=list_colors[i],
                    )
            plt.fill_between(df[['Dimension']].values.ravel(),df[[strat]].values.ravel()- df_std[[strat]].values.ravel(),
                             df[[strat]].values.ravel()+ df_std[[strat]].values.ravel(),color=list_colors[i], alpha=0.2
                            )
        else:
            plt.plot(df[['Dimension']],df[[strat]],linestyle="--")
            plt.scatter(df[['Dimension']],df[[strat]],label=strat)
        
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title,fontsize=fontsize)
    plt.xlabel("Dimension",fontsize=fontsize)
    plt.ylabel("PR AUC",fontsize=fontsize)
    #plt.legend(fontsize=fontsize)
    plt.legend(bbox_to_anchor =(0.5,-0.38),ncol=3, loc='lower center',fontsize=28)
    if to_save:
        plt.savefig(name_file_saving,bbox_inches='tight')
    plt.show()


def compute_asso(X_train,y_train,clf):
    X_train_minority = X_train[y_train == 1]
    preds, gts = [], []
    cv = LeaveOneOut()
    for train, test in cv.split(X_train_minority):
        clf.fit(X_train_minority[train, :-1].astype(float), X_train_minority[train, -1])
        pred = clf.predict(X_train_minority[test, :-1].astype(float))
        preds.append(pred[0])
        gts.append(X_train_minority[test, -1][0])
    gts = np.array(gts)
    essai_preds = np.array(preds)
    res = (gts == essai_preds).mean()
    return res

def compute_asso_over_dimensions(
    dimensions,init_output_dir_path,init_name_file,clf = None,n_iter=20,
):
    list_asso_final_mean = []
    list_asso_final_std = []
    list_acc_bayes = []
    list_acc_bayes_std = []
    list_bayes_diff_mean = []
    list_bayes_diff_std = []
    for i,dimension in enumerate(dimensions):           
        list_asso_dim = []
        list_acc_bayes_curr = []
        ## Bayes classifier :
        X_tilde, target_numeric_tilde, w_gauss_tilde = generate_initial_data_onecat_2025_02_28(dimension_continuous=dimension,n_samples=1000000,random_state=100+i)
        X_minority_tilde = X_tilde[target_numeric_tilde == 1]
        del X_tilde,target_numeric_tilde,w_gauss_tilde
        clf_bayes = lgb.LGBMClassifier(verbosity=-1,n_jobs=5,random_state=0)
        clf_bayes.fit(X_minority_tilde[:, :-1].astype(float), X_minority_tilde[:, -1])
        for j in range(n_iter):
            output_dir_path =init_output_dir_path  +str(dimension)
            name_file = init_name_file + str(j) + ".npy"
            X_train = np.load(
                os.path.join(output_dir_path, "xtrain" + name_file),allow_pickle=True,
            )
            y_train = np.load(
                os.path.join(output_dir_path, "ytrain" + name_file),allow_pickle=True,
            )
            #X_test = np.load(
            #    os.path.join(output_dir_path, "xtest" + name_file),allow_pickle=True,
            #)
            #oversample_strategies = np.load(os.path.join(output_dir_path, "name_strats" + name_file))
            #predictions_by_strategy = np.load(os.path.join(output_dir_path, "preds_" + name_file))
            #df_all = pd.DataFrame(predictions_by_strategy, columns=oversample_strategies)
            #df_fold_0 = df_all[df_all["fold"] == 0] # it's a train/test
            #y_test = df_fold_0[['y_true']].to_numpy().ravel()
            current_asso = compute_asso(X_train,y_train,clf)
            list_asso_dim.append(current_asso)
            #print(current_asso)

            X_train_minority = X_train[y_train==1]
            preds = clf_bayes.predict(X_train_minority[:, :-1].astype(float))
            acc_bayes = (X_train_minority[:, -1] == np.array(preds)).mean()
            list_acc_bayes_curr.append(acc_bayes)

        print("End dim ",  dimension)
        list_asso_final_mean.append(np.mean(list_asso_dim))
        list_asso_final_std.append(np.std(list_asso_dim))
        
        list_acc_bayes.append(np.mean(list_acc_bayes_curr))
        list_acc_bayes_std.append(np.std(list_acc_bayes_curr))
        list_bayes_diff_mean.append(np.mean( np.array(list_acc_bayes_curr) -  np.array(list_asso_dim) ))
        list_bayes_diff_std.append(np.std( np.array(list_acc_bayes_curr) -  np.array(list_asso_dim) ))
        
    print(list_acc_bayes)
    return list_asso_final_mean, list_asso_final_std, list_acc_bayes,list_acc_bayes_std, list_bayes_diff_mean, list_bayes_diff_std