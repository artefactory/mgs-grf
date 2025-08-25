def plot_curves(
    output_dir,
    start_filename,
    n_iter,
    stategies_to_show=None,
    names_stategies_to_show=None,
    show_pr=False,
    show_auc_curves=True,
    to_show=True,
    value_alpha=0.2,
    kind_interpolation="linear",
):
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
    filename_0 = start_filename + str(0) + ".npy"
    if stategies_to_show is None:
        stategies_to_show = np.load(os.path.join(output_dir, "name_strats" + filename_0)).tolist()
        stategies_to_show.remove("fold")  # remove fold column which is not a strategy
        stategies_to_show.remove("y_true")  # remove y_true column which is not a strategy
    if names_stategies_to_show is None:
        names_stategies_to_show = stategies_to_show

    list_names_oversamplings = np.load(os.path.join(output_dir, "name_strats" + filename_0))

    list_fpr = np.arange(start=0, stop=1.01, step=0.01)
    list_recall = np.arange(start=0, stop=1.01, step=0.01)
    array_interpolated_quantity = np.zeros((n_iter, len(list_recall), len(stategies_to_show)))
    array_quantity_auc = np.zeros((n_iter, len(stategies_to_show)))
    for i in range(n_iter):
        filename = start_filename + str(i) + ".npy"
        array_all_preds_strats_final = np.load(os.path.join(output_dir, "preds_" + filename))
        df_all = pd.DataFrame(array_all_preds_strats_final, columns=list_names_oversamplings)

        for j, col in enumerate(stategies_to_show):
            array_interpolated_quantity_folds = np.zeros((5, len(list_recall)))
            list_auc_folds = []
            for fold in range(5):
                df = df_all[df_all["fold"] == fold]
                y_true = df["y_true"].tolist()
                pred_probas_col = df[col].tolist()

                if show_pr:  ## PR Curves case
                    prec, rec, tresh = precision_recall_curve(y_true, pred_probas_col)
                    pr_auc = auc(rec, prec)
                    interpolation_func = interpolate.interp1d(
                        np.flip(rec), np.flip(prec), kind=kind_interpolation
                    )
                    prec_interpolated = interpolation_func(list_recall)
                    # array_interpolated_quantity_folds[fold,:] = prec_interpolated
                    array_interpolated_quantity_folds[fold, :] = np.flip(prec_interpolated)
                    list_auc_folds.append(pr_auc)
                else:  ## ROC Curves case
                    fpr, tpr, _ = roc_curve(y_true, pred_probas_col)
                    interpolation_func = interpolate.interp1d(fpr, tpr, kind=kind_interpolation)
                    tpr_interpolated = interpolation_func(list_fpr)
                    array_interpolated_quantity_folds[fold, :] = tpr_interpolated
                    roc_auc = roc_auc_score(y_true, pred_probas_col)
                    list_auc_folds.append(roc_auc)

            array_interpolated_quantity[i, :, j] = array_interpolated_quantity_folds.mean(
                axis=0
            )  ## the mean interpolated over the 5 fold are averaged
            array_quantity_auc[i, j] = np.mean(list_auc_folds)
    mean_final_prec = array_interpolated_quantity.mean(
        axis=0
    )  ## interpolated precisions over the n_iter ietartions are averaged by strategy
    std_final_prec = array_interpolated_quantity.std(axis=0)
    ########### Plotting curves ##############
    if to_show:
        plt.figure(figsize=(10, 6))
    for h, col in enumerate(names_stategies_to_show):
        if show_pr:  ## PR Curves case
            if show_auc_curves:
                pr_auc_col = auc(np.flip(list_recall), mean_final_prec[:, h])
            else:
                pr_auc_col = array_quantity_auc[:, h].mean()
            lab_col = col + " AUC=" + str(round(pr_auc_col, 3))
            # disp = PrecisionRecallDisplay(precision=mean_final_prec[:,h], recall=np.flip(list_recall))
            # disp.plot()
            plt.plot(np.flip(list_recall), mean_final_prec[:, h], label=lab_col)
            plt.fill_between(
                np.flip(list_recall),
                mean_final_prec[:, h] + std_final_prec[:, h],
                mean_final_prec[:, h] - std_final_prec[:, h],
                alpha=value_alpha,
                step="pre",
            )  # color='grey'
        else:  ## ROC Curves case
            if show_auc_curves:
                pr_auc_col = auc(list_fpr, mean_final_prec[:, h])
            else:
                pr_auc_col = array_quantity_auc[:, h].mean()
            lab_col = col + " AUC=" + str(round(pr_auc_col, 3))
            plt.scatter(list_fpr, mean_final_prec[:, h], label=lab_col)
            plt.fill_between(
                list_fpr,
                mean_final_prec[:, h] + std_final_prec[:, h],
                mean_final_prec[:, h] - std_final_prec[:, h],
                alpha=value_alpha,
                step="pre",
            )  # color='grey'
    #################### Add legend or not (for tuned function ploting) ##################
    if to_show:
        if show_pr:
            plt.legend(loc="best", fontsize="small")
            plt.title("PR Curves", weight="bold", fontsize=15)
            plt.xlabel("Recall", fontsize=12)
            plt.ylabel("Precision", fontsize=12)
        else:
            plt.legend(loc="best", fontsize="small")
            plt.title("ROC Curves", weight="bold", fontsize=15)
            plt.xlabel("False Positive Rate (FPR)", fontsize=12)
            plt.ylabel("True Positive Rate (TPR)", fontsize=12)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.show()


def plot_curves_tuned(
    output_dir,
    start_filename,
    n_iter,
    list_name_strat,
    list_name_strat_inside_file,
    list_name_strat_to_show=None,
    show_pr=False,
    show_auc_curves=True,
    value_alpha=0.2,
    kind_interpolation="linear",
):
    plt.figure(figsize=(10, 6))
    if list_name_strat_to_show is None:
        list_name_strat_to_show = list_name_strat_inside_file
    for i, strat in enumerate(list_name_strat):
        curr_start_output_dir = os.path.join(output_dir, strat, "RF_100")
        plot_curves(
            output_dir=curr_start_output_dir,
            start_filename=start_filename,
            n_iter=n_iter,
            stategies_to_show=[list_name_strat_inside_file[i]],
            names_stategies_to_show=[list_name_strat_to_show[i]],
            show_pr=show_pr,
            show_auc_curves=show_auc_curves,
            to_show=False,
            value_alpha=value_alpha,
            kind_interpolation=kind_interpolation,
        )

    if show_pr:
        plt.legend(loc="best", fontsize="small")
        plt.title("PR Curves", weight="bold", fontsize=15)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
    else:
        plt.legend(loc="best", fontsize="small")
        plt.title("ROC Curves", weight="bold", fontsize=15)
        plt.xlabel("False Positive Rate (FPR)", fontsize=12)
        plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.show()
