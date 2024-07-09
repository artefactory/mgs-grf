"""Short explanation of file."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# Really needs a better name
# my_smote -> specify how it changes from "classical" SMOTE
# sans_init_sample -> more difficult, initless ? initfree ?
def my_smote_sans_init_sample(
    X, n_neighbors, n_final_sample, w_simulation, w_simulation_params
):
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    n_neighbors : _type_
        _description_
    n_final_sample : _type_
        _description_
    w_simulation : _type_
        _description_
    w_simulation_params : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    n_minoritaire = X.shape[0]  # english
    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
    neigh.fit(X)
    neighbor_by_index = neigh.kneighbors(
        X=X, n_neighbors=n_neighbors + 1, return_distance=False
    )  # le premier
    # élément sera tjrs le point en lui-même vu qu'on a fit sur X aussi.
    n_synthetic_samples = n_final_sample  # - n_minoritaire # Why changing name ?
    new_samples = np.zeros((n_synthetic_samples, X.shape[1]))
    for i in range(n_synthetic_samples):
        indice = np.random.randint(n_minoritaire)  # individu centrale qui sera choisi
        w = w_simulation(  # w = weight ? better naming
            w_simulation_params[0], w_simulation_params[1]
        )  # facteur alpha de la première difference
        k1 = np.random.randint(  # k1 = random_neighbor ? better naming
            1, n_neighbors + 1
        )  # Sélection voisin parmi les K. On exclu 0 car c indice lui-même
        indice_neighbor = neighbor_by_index[indice][k1]

        diff_1 = X[indice_neighbor, :] - X[indice, :]  # diff_1 = ? better naming
        new_observation = X[indice, :] + w * diff_1
        new_samples[i, :] = new_observation

    # res_final_X = np.concatenate((X, new_samples), axis=0)
    return new_samples


def compute_mean_dist(X, X_final, n_neighors):
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    X_final : _type_
        _description_
    K : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    neigh = NearestNeighbors(n_neighbors=n_neighors, algorithm="ball_tree")
    neigh.fit(X.reshape(-1, 1))  #### ATTTENTIOn ici avant : X.reshape(-1, 1)
    neighbor_dist, neighbor_by_index = neigh.kneighbors(
        X=X_final, n_neighbors=n_neighors, return_distance=True
    )
    return neighbor_dist[:, 0].mean()


def compute_mean_dist_idee3(
    X, X_final, K=1
):  # Better naming of function, K -> n_neighbors, X_final -> X_synthetic ?
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    X_final : _type_
        _description_
    K : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """
    neigh = NearestNeighbors(n_neighbors=K, algorithm="ball_tree")
    neigh.fit(X)  #### ATTTENTIOn ici avant : X.reshape(-1, 1)
    neighbor_dist, neighbor_by_index = neigh.kneighbors(
        X=X_final, n_neighbors=K, return_distance=True
    )
    return neighbor_dist[:, 0].mean()


############################################################################################
####################### Experiments distance 1 #############################################
############################################################################################
def save_run(
    output_dir,
    name_file,
    list_N,
    list_dist_K5_mean,
    list_dist_Ksqrt_mean,
    list_dist_K0_01_mean,
    list_dist_K01_mean,
    list_dist_K03_mean,
    list_dist_K08_mean,
    list_dist_K5_std,
    list_dist_Ksqrt_std,
    list_dist_K0_01_std,
    list_dist_K01_std,
    list_dist_K03_std,
    list_dist_K08_std,
):
    d_mean = {
        "n": list_N,
        "K=5": list_dist_K5_mean,
        "K=sqrt(n)": list_dist_Ksqrt_mean,
        "K=0.01n": list_dist_K0_01_mean,
        "K=0.1n": list_dist_K01_mean,
        "K=0.3n": list_dist_K03_mean,
        "K=0.8n": list_dist_K08_mean,
    }
    d_std = {
        "n": list_N,
        "K=5": list_dist_K5_std,
        "K=sqrt(n)": list_dist_Ksqrt_std,
        "K=0.01n": list_dist_K0_01_std,
        "K=0.1n": list_dist_K01_std,
        "K=0.3n": list_dist_K03_std,
        "K=0.8n": list_dist_K08_std,
    }

    df_mean = pd.DataFrame(data=d_mean)
    df_std = pd.DataFrame(data=d_std)
    df_mean.to_csv(os.path.join(output_dir, "mean_" + name_file))
    df_std.to_csv(os.path.join(output_dir, "std_" + name_file))


def open_plot_run(
    output_dir_open,
    name_file_open,
    name_y,
    name_title,
    xmin=0,
    xmax=6000,
    savefig=False,
    output_dir_save=None,
    name_file_save=None,
):
    df_mean = pd.read_csv(os.path.join(output_dir_open, "mean_" + name_file_open))
    df_std = pd.read_csv(os.path.join(output_dir_open, "std_" + name_file_open))

    plt.figure(figsize=(10, 8))
    plt.title(name_title)
    plt.errorbar(
        df_mean[["n"]].values.ravel(),  # .values > .to_numpy() everywhere
        df_mean[["K=5"]].values.ravel(),
        yerr=df_std[["K=5"]].values.ravel(),
        label=r"$K=5$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=sqrt(n)"]].values.ravel(),
        yerr=df_std[["K=sqrt(n)"]].values.ravel(),
        label=r"$K=\sqrt{n}$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=0.01n"]].values.ravel(),
        yerr=df_std[["K=0.01n"]].values.ravel(),
        label=r"$K=0.01 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=0.1n"]].values.ravel(),
        yerr=df_std[["K=0.1n"]].values.ravel(),
        label=r"$K=0.1 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=0.3n"]].values.ravel(),
        yerr=df_std[["K=0.3n"]].values.ravel(),
        label=r"$K=0.1 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=0.8n"]].values.ravel(),
        yerr=df_std[["K=0.8n"]].values.ravel(),
        label=r"$K=0.1 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )

    plt.xlabel("Number of initial minority sample (n)", fontsize=23)
    plt.ylabel(name_y, fontsize=32)
    plt.xlim([xmin, xmax])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(title="Legend", fontsize=15)
    if savefig is True:
        plt.savefig(os.path.join(output_dir_save, name_file_save))
    plt.show()


def run_dist_exp(list_N, output_dir, name_file):
    list_dist_K5_mean = []  # no need to put type list_xxx -> xxx
    list_dist_Ksqrt_mean = []
    list_dist_K01_mean = []
    list_dist_K03_mean = []
    list_dist_K08_mean = []
    list_dist_K0_01_mean = []

    list_dist_K5_std = []
    list_dist_Ksqrt_std = []
    list_dist_K01_std = []
    list_dist_K03_std = []
    list_dist_K08_std = []
    list_dist_K0_01_std = []

    for n in list_N:
        current_list_dist_K5 = []
        current_list_dist_Ksqrt = []
        current_list_dist_K01 = []
        current_list_dist_K03 = []
        current_list_dist_K08 = []
        current_list_dist_K0_01 = []
        for i in range(100):
            X = -3 + 6 * np.random.random_sample(n)

            X_final = my_smote_sans_init_sample(
                X=X.reshape(-1, 1),
                K=5,
                n_final_sample=1000,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            current_list_dist_K5.append(compute_mean_dist(X=X, X_final=X_final, K=2))

            X_final = my_smote_sans_init_sample(
                X=X.reshape(-1, 1),
                K=max(1, int(np.sqrt(n))),
                n_final_sample=1000,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            current_list_dist_Ksqrt.append(compute_mean_dist(X=X, X_final=X_final, K=2))

            X_final = my_smote_sans_init_sample(
                X=X.reshape(-1, 1),
                K=max(1, int(0.1 * n)),
                n_final_sample=1000,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            current_list_dist_K01.append(compute_mean_dist(X=X, X_final=X_final, K=2))

            X_final = my_smote_sans_init_sample(
                X=X.reshape(-1, 1),
                K=max(1, int(0.3 * n)),
                n_final_sample=1000,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            current_list_dist_K03.append(compute_mean_dist(X=X, X_final=X_final, K=2))

            X_final = my_smote_sans_init_sample(
                X=X.reshape(-1, 1),
                K=max(1, int(0.8 * n)),
                n_final_sample=1000,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            current_list_dist_K08.append(compute_mean_dist(X=X, X_final=X_final, K=2))

            X_final = my_smote_sans_init_sample(
                X=X.reshape(-1, 1),
                K=max(1, int(0.01 * n)),
                n_final_sample=1000,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            current_list_dist_K0_01.append(compute_mean_dist(X=X, X_final=X_final, K=2))

        list_dist_K5_mean.append(np.mean(current_list_dist_K5))
        list_dist_Ksqrt_mean.append(np.mean(current_list_dist_Ksqrt))
        list_dist_K01_mean.append(np.mean(current_list_dist_K01))
        list_dist_K03_mean.append(np.mean(current_list_dist_K03))
        list_dist_K08_mean.append(np.mean(current_list_dist_K08))
        list_dist_K0_01_mean.append(np.mean(current_list_dist_K0_01))

        list_dist_K5_std.append(np.std(current_list_dist_K5))
        list_dist_Ksqrt_std.append(np.std(current_list_dist_Ksqrt))
        list_dist_K01_std.append(np.std(current_list_dist_K01))
        list_dist_K03_std.append(np.std(current_list_dist_K03))
        list_dist_K08_std.append(np.std(current_list_dist_K08))
        list_dist_K0_01_std.append(np.std(current_list_dist_K0_01))

    save_run(
        output_dir,
        name_file,
        list_N,
        list_dist_K5_mean,
        list_dist_Ksqrt_mean,
        list_dist_K0_01_mean,
        list_dist_K01_mean,
        list_dist_K03_mean,
        list_dist_K08_mean,
        list_dist_K5_std,
        list_dist_Ksqrt_std,
        list_dist_K0_01_std,
        list_dist_K01_std,
        list_dist_K03_std,
        list_dist_K08_std,
    )


############################################################################################
####################### Experiments distance normalized #############################################
############################################################################################


def save_run_normalized(
    output_dir,
    name_file,
    list_N,
    list_dist_K5_mean,
    list_dist_Ksqrt_mean,
    list_dist_K0_01_mean,
    list_dist_K01_mean,
    list_dist_K03_mean,
    list_dist_K08_mean,
    list_normalization_mean,
    list_dist_K5_std,
    list_dist_Ksqrt_std,
    list_dist_K0_01_std,
    list_dist_K01_std,
    list_dist_K03_std,
    list_dist_K08_std,
    list_normalization_std,
):
    """_summary_

    Parameters
    ----------
    output_dir : _type_
        _description_
    name_file : _type_
        _description_
    list_N : _type_
        _description_
    list_dist_K5_mean : _type_
        _description_
    list_dist_Ksqrt_mean : _type_
        _description_
    list_dist_K0_01_mean : _type_
        _description_
    list_dist_K01_mean : _type_
        _description_
    list_dist_K03_mean : _type_
        _description_
    list_dist_K08_mean : _type_
        _description_
    list_normalization_mean : _type_
        _description_
    list_dist_K5_std : _type_
        _description_
    list_dist_Ksqrt_std : _type_
        _description_
    list_dist_K0_01_std : _type_
        _description_
    list_dist_K01_std : _type_
        _description_
    list_dist_K03_std : _type_
        _description_
    list_dist_K08_std : _type_
        _description_
    list_normalization_std : _type_
        _description_
    """
    d_mean = {
        "n": list_N,
        "K=5": list_dist_K5_mean,
        "K=sqrt(n)": list_dist_Ksqrt_mean,
        "K=0.01n": list_dist_K0_01_mean,
        "K=0.1n": list_dist_K01_mean,
        "K=0.3n": list_dist_K03_mean,
        "K=0.8n": list_dist_K08_mean,
        "normalization": list_normalization_mean,
    }
    d_std = {
        "n": list_N,
        "K=5": list_dist_K5_std,
        "K=sqrt(n)": list_dist_Ksqrt_std,
        "K=0.01n": list_dist_K0_01_std,
        "K=0.1n": list_dist_K01_std,
        "K=0.3n": list_dist_K03_std,
        "K=0.8n": list_dist_K08_std,
        "normalization": list_normalization_std,
    }

    df_mean = pd.DataFrame(data=d_mean)
    df_std = pd.DataFrame(data=d_std)
    df_mean.to_csv(os.path.join(output_dir, "mean_" + name_file))
    df_std.to_csv(os.path.join(output_dir, "std_" + name_file))


def open_plot_run_normalized(
    output_dir_open,
    name_file_save,
    name_y,
    title_name,
    xmin=0,
    xmax=6000,
    savefig=False,
    output_dir_save=None,
    name_file_open=None,
):
    """_summary_

    Parameters
    ----------
    output_dir_open : _type_
        _description_
    name_file_save : _type_
        _description_
    name_y : _type_
        _description_
    name_title : _type_
        _description_
    xmin : int, optional
        _description_, by default 0
    xmax : int, optional
        _description_, by default 6000
    bool_to_save : bool, optional
        _description_, by default False
    output_dir_save : _type_, optional
        _description_, by default None
    name_file_open : _type_, optional
        _description_, by default None
    """
    df_mean = pd.read_csv(os.path.join(output_dir_open, "mean_" + name_file_open))
    df_std = pd.read_csv(os.path.join(output_dir_open, "std_" + name_file_open))

    plt.figure(figsize=(10, 8))
    plt.title(title_name)
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=5"]].values.ravel() / df_mean[["normalization"]].values.ravel(),
        yerr=df_std[["K=5"]].values.ravel(),
        label=r"$K=5$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=sqrt(n)"]].values.ravel()
        / df_mean[["normalization"]].values.ravel(),
        yerr=df_std[["K=sqrt(n)"]].values.ravel(),
        label=r"$K=\sqrt{n}$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=0.01n"]].values.ravel() / df_mean[["normalization"]].values.ravel(),
        yerr=df_std[["K=0.01n"]].values.ravel(),
        label=r"$K=0.01 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=0.1n"]].values.ravel() / df_mean[["normalization"]].values.ravel(),
        yerr=df_std[["K=0.1n"]].values.ravel(),
        label=r"$K=0.1 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=0.3n"]].values.ravel() / df_mean[["normalization"]].values.ravel(),
        yerr=df_std[["K=0.3n"]].values.ravel(),
        label=r"$K=0.1 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].values.ravel(),
        df_mean[["K=0.8n"]].values.ravel() / df_mean[["normalization"]].values.ravel(),
        yerr=df_std[["K=0.8n"]].values.ravel(),
        label=r"$K=0.1 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )

    plt.xlabel("Number of initial minority sample (n)", fontsize=23)
    plt.ylabel(name_y, fontsize=32)
    plt.xlim([xmin, xmax])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(title="Legend", fontsize=15)
    if savefig is True:
        plt.savefig(os.path.join(output_dir_save, name_file_save))
    plt.show()


def run_dist_exp_normalized(list_N, output_dir, name_file):
    list_dist_K5_mean = []
    list_dist_Ksqrt_mean = []
    list_dist_K01_mean = []
    list_dist_K03_mean = []
    list_dist_K08_mean = []
    list_dist_K0_01_mean = []

    list_dist_K5_std = []
    list_dist_Ksqrt_std = []
    list_dist_K01_std = []
    list_dist_K03_std = []
    list_dist_K08_std = []
    list_dist_K0_01_std = []

    list_normalization_mean = []
    list_normalization_std = []

    m = 1000

    for n in list_N:
        current_list_dist_K5 = []
        current_list_dist_Ksqrt = []
        current_list_dist_K01 = []
        current_list_dist_K03 = []
        current_list_dist_K08 = []
        current_list_dist_K0_01 = []
        current_list_normalization = []
        for i in range(75):
            X_original = np.random.uniform(low=(-3, -3), high=(3, 3), size=(n, 2))
            X_original_tmp = np.random.uniform(low=(-3, -3), high=(3, 3), size=(m, 2))
            X_K5 = my_smote_sans_init_sample(
                X=X_original,
                K=5,
                n_final_sample=m,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            X_Ksqrt = my_smote_sans_init_sample(
                X=X_original,
                K=int(np.sqrt(n)),
                n_final_sample=m,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            X_K01 = my_smote_sans_init_sample(
                X=X_original,
                K=int(0.1 * n),
                n_final_sample=m,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            X_K03 = my_smote_sans_init_sample(
                X=X_original,
                K=int(0.3 * n),
                n_final_sample=m,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            X_K08 = my_smote_sans_init_sample(
                X=X_original,
                K=int(0.8 * n),
                n_final_sample=m,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )
            X_K0_01 = my_smote_sans_init_sample(
                X=X_original,
                K=max(int(0.01 * n), 1),
                n_final_sample=m,
                w_simulation=np.random.uniform,
                w_simulation_params=[0, 1],
            )

            current_list_normalization.append(
                compute_mean_dist_idee3(X=X_original, X_final=X_original_tmp, K=1)
            )
            current_list_dist_K5.append(
                compute_mean_dist_idee3(X=X_original, X_final=X_K5, K=1)
            )
            current_list_dist_Ksqrt.append(
                compute_mean_dist_idee3(X=X_original, X_final=X_Ksqrt, K=1)
            )
            current_list_dist_K01.append(
                compute_mean_dist_idee3(X=X_original, X_final=X_K01, K=1)
            )
            current_list_dist_K03.append(
                compute_mean_dist_idee3(X=X_original, X_final=X_K03, K=1)
            )
            current_list_dist_K08.append(
                compute_mean_dist_idee3(X=X_original, X_final=X_K08, K=1)
            )
            current_list_dist_K0_01.append(
                compute_mean_dist_idee3(X=X_original, X_final=X_K0_01, K=1)
            )

        list_dist_K5_mean.append(np.mean(current_list_dist_K5))
        list_dist_Ksqrt_mean.append(np.mean(current_list_dist_Ksqrt))
        list_dist_K01_mean.append(np.mean(current_list_dist_K01))
        list_dist_K03_mean.append(np.mean(current_list_dist_K03))
        list_dist_K08_mean.append(np.mean(current_list_dist_K08))
        list_dist_K0_01_mean.append(np.mean(current_list_dist_K0_01))
        list_normalization_mean.append(np.mean(current_list_normalization))

        list_dist_K5_std.append(np.std(current_list_dist_K5))
        list_dist_Ksqrt_std.append(np.std(current_list_dist_Ksqrt))
        list_dist_K01_std.append(np.std(current_list_dist_K01))
        list_dist_K03_std.append(np.std(current_list_dist_K03))
        list_dist_K08_std.append(np.std(current_list_dist_K08))
        list_dist_K0_01_std.append(np.std(current_list_dist_K0_01))
        list_normalization_std.append(np.std(current_list_normalization))

    save_run_normalized(
        output_dir,
        name_file,
        list_N,
        list_dist_K5_mean,
        list_dist_Ksqrt_mean,
        list_dist_K0_01_mean,
        list_dist_K01_mean,
        list_dist_K03_mean,
        list_dist_K08_mean,
        list_normalization_mean,
        list_dist_K5_std,
        list_dist_Ksqrt_std,
        list_dist_K0_01_std,
        list_dist_K01_std,
        list_dist_K03_std,
        list_dist_K08_std,
        list_normalization_std,
    )


############################################################################################
############################## Données réelles #############################################
############################################################################################


def ROS_init_sample(X, n_final_sample):
    n_minoritaire = X.shape[0]
    n_synthetic_sample = n_final_sample  # - n_minoritaire # useless --> remove
    new_samples = np.zeros((n_synthetic_sample, X.shape[1]))

    for i in range(n_synthetic_sample):
        indice = np.random.randint(n_minoritaire)  # individu choisi
        new_samples[i, :] = X[indice, :]

    return new_samples


def no_sampling_init(X, n_final_sample):
    return X


def protocole_dist_mean_dataset(df_positifs, n_iter=1):
    n_samples = len(df_positifs)
    list_oversampling_and_params = [
        (
            my_smote_sans_init_sample,
            {"w_simulation": np.random.uniform, "w_simulation_params": [0, 1], "K": 5},
            "SMOTE K=5",
        ),
        (
            my_smote_sans_init_sample,
            {
                "w_simulation": np.random.uniform,
                "w_simulation_params": [0, 1],
                "K": max(int(np.sqrt(round(n_samples / 2))), 1),
            },
            "SMOTE K=sqrt(5)",
        ),
        (
            my_smote_sans_init_sample,
            {
                "w_simulation": np.random.uniform,
                "w_simulation_params": [0, 1],
                "K": max(int(0.01 * round(n_samples / 2)), 1),
            },
            "SMOTE K=0.01n",
        ),
        (
            my_smote_sans_init_sample,
            {
                "w_simulation": np.random.uniform,
                "w_simulation_params": [0, 1],
                "K": max(int(0.1 * round(n_samples / 2)), 1),
            },
            "SMOTE K=0.1n",
        ),
        (
            my_smote_sans_init_sample,
            {
                "w_simulation": np.random.uniform,
                "w_simulation_params": [0, 1],
                "K": max(int(0.3 * round(n_samples / 2)), 1),
            },
            "SMOTE K=0.3n",
        ),
        (
            my_smote_sans_init_sample,
            {
                "w_simulation": np.random.uniform,
                "w_simulation_params": [0, 1],
                "K": max(int(0.8 * round(n_samples / 2)), 1),
            },
            "SMOTE K=0.8n",
        ),
        (ROS_init_sample, {}, "ROS"),
    ]
    n_strategy = len(list_oversampling_and_params)
    # list_all_dist = [[] for i in range(n_strategy)]

    current_list_all_dist = [[] for i in range(n_strategy)]
    for j in range(n_iter):
        df_shuffled = df_positifs.sample(frac=1)
        list_two_split = np.array_split(df_shuffled, 2)
        X_sample_observed = list_two_split[0]
        X_sample_unobserved = list_two_split[1]
        n_final_sample = len(X_sample_observed)

        list_oversampling_func, list_oversampling_params, list_oversampling_name = zip(
            *list_oversampling_and_params
        )
        for i, (oversampling_func, oversampling_params, oversampling_name) in enumerate(
            zip(
                list_oversampling_func, list_oversampling_params, list_oversampling_name
            )
        ):
            X_res = oversampling_func(
                X=X_sample_observed.to_numpy(),
                n_final_sample=n_final_sample,
                **oversampling_params,
            )
            value_strategy = compute_mean_dist_idee3(
                X=X_sample_observed.to_numpy(), X_final=X_res
            )
            value_normalization = compute_mean_dist_idee3(
                X=X_sample_observed.to_numpy(), X_final=X_sample_unobserved.to_numpy()
            )
            value_normalized = value_strategy / value_normalization
            current_list_all_dist[i].append(value_normalized)

    values_mean = np.mean(np.array(current_list_all_dist), axis=1)
    values_std = np.std(np.array(current_list_all_dist), axis=1)
    return values_mean, values_std


def protocole_dist_mean_dataset_onedataset(df_positifs, list_n, n_iter=100, n_iter_2=1):
    list_res_n_mean = []
    list_res_n_std = []
    for n in list_n:
        list_all_dist = [[] for i in range(n_iter)]

        for i in range(n_iter):
            df_positifs_sample = df_positifs.sample(n=n, axis=0)
            values_mean, values_std = protocole_dist_mean_dataset(
                df_positifs_sample, n_iter=n_iter_2
            )
            list_all_dist[i].append(values_mean)

        res_mean = np.mean(
            np.array(list_all_dist), axis=0
        )  # axis =0 car c'est une liste d'array
        res_std = np.std(np.array(list_all_dist), axis=0)  ## et non une liste de liste.
        list_res_n_mean.append(res_mean)
        list_res_n_std.append(res_std)

    return list_res_n_mean, list_res_n_std


def save_mutiple_run_real_data(
    output_dir, name_file, list_res_n_mean, list_res_n_std, list_n
):
    d_mean = np.mean(
        np.array(list_res_n_mean), axis=1
    )  # transforme les données en array
    d_mean = np.hstack((np.array(list_n).reshape(-1, 1), d_mean))  # ajout de list_n
    d_std = np.mean(np.array(list_res_n_std), axis=1)
    d_std = np.hstack((np.array(list_n).reshape(-1, 1), d_std))

    df_mean = pd.DataFrame(
        d_mean,
        columns=[
            "n",
            "K=5",
            "K=sqrt(n)",
            "K=0.01n",
            "K=0.1n",
            "K=0.3n",
            "K=0.8n",
            "ROS",
        ],
    )
    df_std = pd.DataFrame(
        d_std,
        columns=[
            "n",
            "K=5",
            "K=sqrt(n)",
            "K=0.01n",
            "K=0.1n",
            "K=0.3n",
            "K=0.8n",
            "ROS",
        ],
    )
    df_mean.to_csv(os.path.join(output_dir, "mean_" + name_file))
    df_std.to_csv(os.path.join(output_dir, "std_" + name_file))
    # return df_mean,df_std


def open_mutiple_run_real_data(output_dir, name_file):
    df_mean = pd.read_csv(os.path.join(output_dir, "mean_" + name_file), header=0)
    df_std = pd.read_csv(os.path.join(output_dir, "std_" + name_file), header=0)
    return df_mean, df_std


def plot_mutiple_run_real_data(
    df_mean,
    df_std,
    name_y,
    name_title,
    xmin,
    xmax,
    bool_to_save=False,
    output_dir_save=None,
    name_file_save=None,
):
    plt.figure(figsize=(10, 8))
    plt.title(name_title, fontsize=20)
    plt.errorbar(
        df_mean[["n"]].to_numpy().ravel() / 2,
        df_mean[["K=5"]].to_numpy().ravel(),
        yerr=df_std[["K=5"]].to_numpy().ravel(),
        label=r"$K=5$",
        fmt="-o",
        ecolor="black",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].to_numpy().ravel() / 2,
        df_mean[["K=sqrt(n)"]].to_numpy().ravel(),
        yerr=df_std[["K=sqrt(n)"]].to_numpy().ravel(),
        label=r"$K=\sqrt{n}$",
        fmt="-o",
        ecolor="black",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].to_numpy().ravel() / 2,
        df_mean[["K=0.01n"]].to_numpy().ravel(),
        yerr=df_std[["K=0.01n"]].to_numpy().ravel(),
        label=r"$K=0.01 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].to_numpy().ravel() / 2,
        df_mean[["K=0.1n"]].to_numpy().ravel(),
        yerr=df_std[["K=0.1n"]].to_numpy().ravel(),
        label=r"$K=0.1 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].to_numpy().ravel() / 2,
        df_mean[["K=0.3n"]].to_numpy().ravel(),
        yerr=df_std[["K=0.3n"]].to_numpy().ravel(),
        label=r"$K=0.3 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )
    plt.errorbar(
        df_mean[["n"]].to_numpy().ravel() / 2,
        df_mean[["K=0.8n"]].to_numpy().ravel(),
        yerr=df_std[["K=0.8n"]].to_numpy().ravel(),
        label=r"$K=0.8 \times n$",
        fmt="-o",
        ecolor="lightgray",
        elinewidth=2,
        capsize=6,
    )

    plt.xlabel("Number of initial minority sample (n)", fontsize=23)
    plt.ylabel(name_y, fontsize=25)
    plt.xlim([xmin, xmax])
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.legend(title="Legend", fontsize=14)
    if bool_to_save is True:
        plt.savefig(os.path.join(output_dir_save, name_file_save))
    plt.show()
