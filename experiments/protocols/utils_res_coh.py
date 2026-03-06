"""
This file contains the functions to compute the association metric and to plot it over dimensions.
The functions are specifically designed for being imported by the notebook experiments/protocols/notebook/res_coh.ipynb.
"""
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score,balanced_accuracy_score,average_precision_score


from experiments.validation.classif_experiments import (compute_metrics, compute_metrics_several_protocols,
                                            prec_at_recall_version3, prec_at_recall_version3_02)

list_metric = [
    (prec_at_recall_version3_02,'p@r=0.2','proba'),
    (prec_at_recall_version3,'p@r=0.5','proba'),
    (average_precision_score,'avg_pr_auc', 'proba'),
    (roc_auc_score,'roc_auc','proba') 
]

def compute_metric_and_incoherent(output_dir,list_metric,curr_metric= "avg_pr_auc",strategy_name = "SmoteNC (K=5)",
                                  n_iter=20,categorical_features= [-2,-1],init_name_file="2027-01-07-lgbm_"):

    list_metric_none=[]
    list_metric_strategy=[]
    list_ncso=[]
    list_ncss=[]
    for i in range(n_iter):
        name_file=init_name_file+str(i)+".npy"
        df_final_mean,df_final_std =compute_metrics(output_dir=output_dir,name_file=name_file,list_metric=list_metric,n_fold=1)
        strategy_value = df_final_mean.loc[[curr_metric],[strategy_name]].to_numpy().ravel()[0]
        none_value = df_final_mean.loc[[curr_metric],['None']].to_numpy().ravel()[0]
        list_metric_none.append(none_value)
        list_metric_strategy.append(strategy_value)
        
        X_train = np.load(os.path.join(output_dir, "xtrain" + name_file),allow_pickle=True)
        y_train = np.load(os.path.join(output_dir, "ytrain" + name_file))
        X_res = np.load(os.path.join(output_dir, "xres" +strategy_name+ name_file),allow_pickle=True)
        y_res = np.load(os.path.join(output_dir, "yres" +strategy_name+ name_file),allow_pickle=True)
        X_train_minority_comb = [str(xxx) for xxx in X_train[y_train==1][:,categorical_features]]
        counter_train = Counter(X_train_minority_comb)
        X_res_minority_comb = [str(xxx) for xxx in X_res[y_res==1][:,categorical_features]]
        counter_res = Counter(X_res_minority_comb)
    
        ncso = 0
        train_keys = counter_train.keys()
        for key in counter_res.keys():
            if key not in train_keys:
                ncso += counter_res[key]
        list_ncso.append(ncso/ sum(counter_res.values())) ## Compute and save ncso
        
        ncss = 0
        n_original = sum(counter_train.values())
        train_keys = counter_train.keys()
        for key in counter_res.keys():
            if key not in train_keys:
                ncss += counter_res[key]
        
        list_ncss.append(ncss / (sum(counter_res.values())-n_original) ) ## Compute and save ncss
    return list_metric_none, list_metric_strategy, list_ncso, list_ncss


def average_runing_time(output_dir_path,init_name_file,n_iter=50):
    df_final = pd.DataFrame()
    for i in range(n_iter):
        name_file = init_name_file + str(i) + ".csv"
        curr_df = pd.read_csv(os.path.join(output_dir_path, "runtime" + name_file))
        df_final = pd.concat([df_final,curr_df],axis=0)
    return df_final.reset_index(drop=True)


def compute_runs_over_samp(list_samps,strategies,init_output_dir_path,init_name_file,n_iter=20,n_fold=5):
    array_res_mean = np.zeros((len(list_samps),len(strategies)))
    array_res_std = np.zeros((len(list_samps),len(strategies)))
    
    for i,samp in enumerate(list_samps) :
        output_dir_path =init_output_dir_path +str(samp)
        df_final_mean,df_final_std = compute_metrics_several_protocols(
            output_dir=output_dir_path,
            init_name_file=init_name_file,
            list_metric=list_metric,
            bool_roc_auc_only=False,n_iter=n_iter,n_fold=n_fold)
        for j,strat in enumerate(strategies) :
            array_res_mean[i,j] = df_final_mean.loc['avg_pr_auc'][strat]
            array_res_std[i,j] = df_final_std.loc['avg_pr_auc'][strat]
    
    array_res_mean_with_dim = np.hstack((np.array(list_samps).reshape(-1,1),array_res_mean))
    array_res_std_with_dim = np.hstack((np.array(list_samps).reshape(-1,1),array_res_std))
    columns_with_dim = ['n_samples_min']
    columns_with_dim.extend(strategies)
    df_res_mean= pd.DataFrame(array_res_mean_with_dim,columns=columns_with_dim)
    df_res_std= pd.DataFrame(array_res_std_with_dim,columns=columns_with_dim)
    return df_res_mean,df_res_std
    
def plot_(df,df_std,xlim=[-0.1,510],ylim=[0.55,1.0],fontsize=20,plot_error_fill=False,title='',
         name_strats_to_plot= None,to_save=False,name_file_saving='img.pdf'):
    plt.figure(figsize=(12,8))
    list_start = df.columns.tolist()[1:]
    list_fmt= ['o','v','^','s','*','8']*((len(list_start) // 6)+1)
    list_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                   '#7f7f7f', '#bcbd22', '#17becf'] *((len(list_start) // 10)+1)
    for i,strat in enumerate(list_start):
        if plot_error_fill:
            if name_strats_to_plot :
                name_strat= name_strats_to_plot[i]
            else:
                name_strat=strat
            #plt.errorbar(df[['n_samples_min']].values.ravel(),df[[strat]].values.ravel(),
            #                 yerr=df_std[[strat]].values.ravel(),fmt=list_fmt[i], markersize=3,elinewidth=0.5, capsize=6,label=name_strat
            #                )
            plt.plot(df[['n_samples_min']].values.ravel(),df[[strat]].values.ravel(), 
                     marker=list_fmt[i],linestyle="--", label=name_strat,c=list_colors[i],
                    )
            plt.fill_between(df[['n_samples_min']].values.ravel(),df[[strat]].values.ravel()- df_std[[strat]].values.ravel(),
                             df[[strat]].values.ravel()+ df_std[[strat]].values.ravel(),color=list_colors[i], alpha=0.2
                            )
        else:
            plt.plot(df[['n_samples_min']],df[[strat]],linestyle="--")
            plt.scatter(df[['n_samples_min']],df[[strat]],label=strat)

    plt.axvline(x = 3600, color = 'gray',alpha=0.8, label = 'Equilibrium')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title,fontsize=fontsize)
    plt.xlabel("n final",fontsize=fontsize)
    plt.ylabel("PR AUC",fontsize=fontsize)
    #plt.legend(fontsize=fontsize)
    plt.legend(bbox_to_anchor =(0.5,-0.38),ncol=3, loc='lower center',fontsize=fontsize)
    if to_save:
        plt.savefig(name_file_saving,bbox_inches='tight')
    plt.show()