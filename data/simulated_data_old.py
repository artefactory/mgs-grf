import numpy as np
from collections import Counter
from scipy.io.arff import loadarff

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


def my_log_reg(x, beta=np.array([-8, 7, 6]), intercept=-2):
    # beta = np.array([-8,7,6])

    tmp = x.dot(beta)
    z = tmp + intercept  # add intercept
    res = np.exp(z) / (1 + np.exp(z))
    return res


def proba_to_label(y_pred_probas, treshold=0.5):  # apply_threshold ?
    return np.array(np.array(y_pred_probas) >= treshold, dtype=int)


import matplotlib.pyplot as plt


def generate_synthetic_features_logreg(
    X,
    index_informatives,
    list_modalities=["A", "B"],
    beta=np.array([-8, 7, 6]),
    intercept=-2,
    treshold=0.5,
):
    res_log_reg = my_log_reg(X[:, index_informatives], beta=beta, intercept=intercept)
    # plt.hist(res_log_reg)
    # plt.xlim([-0.05,1.05])
    # plt.title(r'y_pred_proba histogram for dimension=%i',fontsize=10)
    # plt.show()

    # pred_logreg = proba_to_label(y_pred_probas=res_log_reg, treshold=treshold)
    # array_final = np.char.replace(pred_logreg.astype(str), '0',list_modalities[0])
    # array_final = np.char.replace(array_final.astype(str), '1',list_modalities[1])
    # return array_final, pred_logreg
    real_pred_log = np.random.binomial(n=1, p=res_log_reg)
    array_final = np.char.replace(real_pred_log.astype(str), "0", list_modalities[0])
    array_final = np.char.replace(array_final.astype(str), "1", list_modalities[1])
    return array_final, real_pred_log


def generate_initial_data_onecat(dimension, n_samples, random_state=24, verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0, high=1, size=(n_samples, 1))
    for i in range(dimension - 2):
        np.random.seed(seed=random_state + i + 1)
        curr_covariate = np.random.uniform(low=0, high=1, size=(n_samples, 1))
        Xf = np.hstack((Xf, curr_covariate))

    ### Feature categorical
    feature_cat_uniform, feature_cat_uniform_numeric = (
        generate_synthetic_features_logreg(
            X=Xf,
            index_informatives=[0, 1, 2],
            list_modalities=["C", "D"],
            beta=np.array([-8, 7, 6]),
            intercept=-2,
        )
    )
    if verbose == 0:
        print("Composition of categorical feature : ", Counter(feature_cat_uniform))
    X_final = np.hstack((Xf, feature_cat_uniform_numeric.reshape(-1, 1)))
    target, target_numeric = generate_synthetic_features_logreg(
        X=X_final,
        index_informatives=[0, 1, 2, -1],
        list_modalities=["No", "Yes"],
        beta=[4, -3, -3, 3],
        intercept=-3.5,
    )
    if verbose == 0:
        print("Composition of the target ", Counter(target))
    return X_final, target, target_numeric


def generate_initial_data_onecat_normal(
    dimension, n_samples, random_state=24, verbose=0
):
    np.random.seed(random_state)
    Xf = np.random.multivariate_normal(
        mean=np.zeros((dimension - 1,)),
        cov=np.eye(dimension - 1, dimension - 1),
        size=n_samples,
    )
    ### Feature categorical
    feature_cat_uniform, feature_cat_uniform_numeric = (
        generate_synthetic_features_logreg(
            X=Xf, index_informatives=[0, 1, 2], list_modalities=["C", "D"]
        )
    )

    X_final = np.hstack((Xf, feature_cat_uniform_numeric.reshape(-1, 1)))
    target, target_numeric = generate_synthetic_features_logreg(
        X=X_final,
        index_informatives=[0, 1, 2, -1],
        list_modalities=["No", "Yes"],
        beta=[-7, 4, 6, 7],
        intercept=-11,
        treshold=0.5,
    )
    if verbose == 0:
        print("Composition of the target before subsampling ", Counter(target_numeric))
    X_final, target_numeric = subsample_to_ratio(
        X=X_final, y=target_numeric, ratio=0.08, seed_sub=random_state
    )
    if verbose == 0:
        print("Composition of the target ", Counter(target_numeric))
        print("Composition of categorical feature : ", Counter(feature_cat_uniform))
    return X_final, target_numeric, target_numeric


def generate_synthetic_features_logreg_triple(
    X,
    index_informatives,
    list_modalities=["A", "B", "C"],
    beta1=np.array([-8, 7, 6]),
    beta2=np.array([8, -7, 6]),
    beta3=np.array([8, 7, -6]),
):
    res_log_reg1 = my_log_reg(X[:, index_informatives], beta=beta1)
    res_log_reg2 = my_log_reg(X[:, index_informatives], beta=beta2)
    res_log_reg3 = my_log_reg(X[:, index_informatives], beta=beta3)
    res_log_reg = np.hstack(
        (
            res_log_reg1.reshape(-1, 1),
            res_log_reg2.reshape(-1, 1),
            res_log_reg3.reshape(-1, 1),
        )
    )
    array_argmax = np.argmax(res_log_reg, axis=1)

    array_final = np.char.replace(array_argmax.astype(str), "0", list_modalities[0])
    array_final = np.char.replace(array_final.astype(str), "1", list_modalities[1])
    array_final = np.char.replace(array_final.astype(str), "2", list_modalities[2])
    return array_final, array_argmax


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


def generate_synthetic_features_logreg_2(
    X,
    index_informatives,
    list_modalities=["A", "B"],
    beta=np.array([-8, 7, 6]),
    intercept=-2,
):
    res_log_reg = my_log_reg(X[:, index_informatives], beta=beta, intercept=-2)
    return res_log_reg


def generate_initial_data_twocat_fit(dimension, n_samples, random_state=123, verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0, high=1, size=(n_samples, 1))
    for i in range(dimension - 3):
        np.random.seed(seed=random_state + i + 1)
        curr_covariate = np.random.uniform(low=0, high=1, size=(n_samples, 1))
        Xf = np.hstack((Xf, curr_covariate))

    feature_cat_uniform, feature_cat_uniform_numeric = (
        generate_synthetic_features_logreg_triple(
            X=Xf,
            index_informatives=[0, 1, 2],
            list_modalities=["A", "B", "C"],
            beta1=np.array([-8, 7, 6]),
            beta2=np.array([4, -7, 3]),
            beta3=np.array([2, -1, 2]),
        )
    )
    X_final = np.hstack((Xf, feature_cat_uniform.reshape(-1, 1)))
    X_final_num = np.hstack((Xf, feature_cat_uniform_numeric.reshape(-1, 1)))
    feature_cat_uniform2, feature_cat_uniform_numeric2 = (
        generate_synthetic_features_logreg_triple(
            X=Xf,
            index_informatives=[0, 1, 2],
            list_modalities=["D", "E", "F"],
            beta1=np.array([-4, 5, 6]),
            beta2=np.array([6, -3, 2]),
            beta3=np.array([1, 5, -1]),
        )
    )
    X_final = np.hstack((X_final, feature_cat_uniform2.reshape(-1, 1)))
    X_final_num = np.hstack((X_final_num, feature_cat_uniform_numeric2.reshape(-1, 1)))

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_final_cat_enc = enc.fit_transform(X_final[:, -2:])
    print(enc.get_feature_names_out())
    X_final_enc = np.hstack((Xf, X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0, 1, 2]
    list_index_informatives.extend([-(i + 1) for i in range(n_modalities)])
    beta = [11, -8.1, -9, -1, 8, 5, -3, -5, 2, 6]
    print(list_index_informatives)
    print(beta[: (n_modalities + 3)])
    print(X_final_enc[:, list_index_informatives].shape)

    probas = generate_synthetic_features_logreg_2(
        X=X_final_enc,
        index_informatives=list_index_informatives,
        list_modalities=["No", "Yes"],
        beta=beta[: (n_modalities + 3)],
        intercept=-5,
    )
    # y_num = np.random.normal(loc=0.5,scale=0.25,size=n_samples)
    rf = RandomForestRegressor(random_state=1234, max_depth=None)
    rf.fit(X_final_enc[:, list_index_informatives], probas)
    print(
        "Mean depth : ",
        np.mean([rf.estimators_[i].get_depth() for i in range(len(rf.estimators_))]),
    )
    return rf, enc


def generate_initial_data_twocat(
    dimension, n_samples, rf, enc, random_state=24, verbose=0
):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0, high=1, size=(n_samples, 1))
    for i in range(dimension - 3):
        np.random.seed(seed=random_state + i + 1)
        curr_covariate = np.random.uniform(low=0, high=1, size=(n_samples, 1))
        Xf = np.hstack((Xf, curr_covariate))

    feature_cat_uniform, feature_cat_uniform_numeric = (
        generate_synthetic_features_logreg_triple(
            X=Xf,
            index_informatives=[0, 1, 2],
            list_modalities=["A", "B", "C"],
            beta1=np.array([-8, 7, 6]),
            beta2=np.array([4, -7, 3]),
            beta3=np.array([2, -1, 2]),
        )
    )
    X_final = np.hstack((Xf, feature_cat_uniform.reshape(-1, 1)))
    X_final_num = np.hstack((Xf, feature_cat_uniform_numeric.reshape(-1, 1)))
    feature_cat_uniform2, feature_cat_uniform_numeric2 = (
        generate_synthetic_features_logreg_triple(
            X=Xf,
            index_informatives=[0, 1, 2],
            list_modalities=["D", "E", "F"],
            beta1=np.array([-4, 5, 6]),
            beta2=np.array([6, -3, 2]),
            beta3=np.array([1, 5, -1]),
        )
    )
    X_final = np.hstack((X_final, feature_cat_uniform2.reshape(-1, 1)))
    X_final_num = np.hstack((X_final_num, feature_cat_uniform_numeric2.reshape(-1, 1)))

    X_final_cat_enc = enc.transform(X_final[:, -2:])
    X_final_enc = np.hstack((Xf, X_final_cat_enc))
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0, 1, 2]
    list_index_informatives.extend([-(i + 1) for i in range(n_modalities)])

    target_proba = rf.predict(X_final_enc[:, list_index_informatives])
    target = proba_to_label(y_pred_probas=target_proba, treshold=0.8)
    if verbose == 1:
        print("Composition of the target ", Counter(target))
        print("Composition of categorical feature 1 : ", Counter(feature_cat_uniform))
        print("Composition of categorical feature 2 : ", Counter(feature_cat_uniform2))
        print("###########")
        print("Composition of the target ", Counter(target))
        print("Composition of categorical feature 1 : ", Counter(feature_cat_uniform))
        print("Composition of categorical feature 2 : ", Counter(feature_cat_uniform2))
        print("***************")

    X_final, target = subsample_to_ratio(
        X=X_final, y=target, ratio=0.1, seed_sub=random_state
    )

    if verbose == 2:
        print("Composition of the target ", Counter(target))
        print("Composition of categorical feature 1 : ", Counter(feature_cat_uniform))
        print("Composition of categorical feature 2 : ", Counter(feature_cat_uniform2))
        print("###########")

    return X_final, target, target


def generate_synthetic_features_multinomial_quadruple(
    X,
    index_informatives,
    list_modalities=["A", "B", "C", "D"],
    beta1=np.array([-8, 7, 6]),
    beta2=np.array([8, -7, 6]),
    beta3=np.array([8, 7, -6]),
    beta4=np.array([-3, -2, 8]),
    intercept=-2,
):
    linear1 = my_log_reg(X[:, index_informatives], beta=beta1, intercept=intercept)
    linear2 = my_log_reg(X[:, index_informatives], beta=beta2, intercept=intercept)
    linear3 = my_log_reg(X[:, index_informatives], beta=beta3, intercept=intercept)
    linear4 = my_log_reg(X[:, index_informatives], beta=beta4, intercept=intercept)
    sum_linear = linear1 + linear2 + linear3
    probas1 = linear1 / (1 + sum_linear)
    probas2 = linear2 / (1 + sum_linear)
    probas3 = linear3 / (1 + sum_linear)
    probas4 = 1 / (1 + sum_linear)
    # print('Somme des 4 probas : ', probas1+probas2+probas3+probas4)
    array_probas = np.array((probas1, probas2, probas3, probas4)).T
    # print('array_probas shape :', array_probas.shape)

    pred_log = np.array(
        [
            np.random.multinomial(n=1, pvals=array_probas[i, :])
            for i in range(len(array_probas))
        ]
    )
    # pred_log = np.random.multinomial(n=1,pvals=array_probas)
    # pred_log = np.apply_along_axis(np.random.multinomial,1,array_probas,{'n':1})
    # print('pred_log shape : ',pred_log.shape)
    # print('pred_log',pred_log)

    array_argmax = np.argmax(pred_log, axis=1)
    # print('array_argmax shape : ',array_argmax.shape)
    # print('array_argmax',array_argmax)

    array_final = np.char.replace(array_argmax.astype(str), "0", list_modalities[0])
    array_final = np.char.replace(array_final.astype(str), "1", list_modalities[1])
    array_final = np.char.replace(array_final.astype(str), "2", list_modalities[2])
    array_final = np.char.replace(array_final.astype(str), "3", list_modalities[3])
    # print('array_final shape ',array_final)


def generate_synthetic_features_multinomial_nonuple(
    X,
    index_informatives,
    list_modalities=["Ae", "Bd", "Af", "Ce", "Ad", "Be", "Bf", "Cd", "Cf"],
    beta1=np.array([-8, 7, 6]),
    beta2=np.array([8, -7, 6]),
    beta3=np.array([8, 7, -6]),
    beta4=np.array([-3, -2, 8]),
    beta5=np.array([2, 5, 1]),
    beta6=np.array([3, 2, 8]),
    beta7=np.array([7, 6, 6]),
    beta8=np.array([1, 3, 1]),
    beta9=np.array([1, 1, 9]),
    intercept=-2,
):
    linear1 = my_log_reg(X[:, index_informatives], beta=beta1, intercept=intercept)
    linear2 = my_log_reg(X[:, index_informatives], beta=beta2, intercept=intercept)
    linear3 = my_log_reg(X[:, index_informatives], beta=beta3, intercept=intercept)
    linear4 = my_log_reg(X[:, index_informatives], beta=beta4, intercept=intercept)
    linear5 = my_log_reg(X[:, index_informatives], beta=beta5, intercept=intercept)
    linear6 = my_log_reg(X[:, index_informatives], beta=beta6, intercept=intercept)
    linear7 = my_log_reg(X[:, index_informatives], beta=beta7, intercept=intercept)
    linear8 = my_log_reg(X[:, index_informatives], beta=beta8, intercept=intercept)
    sum_linear = (
        linear1 + linear2 + linear3 + linear4 + linear5 + linear6 + linear7 + linear8
    )
    probas1 = linear1 / (1 + sum_linear)
    probas2 = linear2 / (1 + sum_linear)
    probas3 = linear3 / (1 + sum_linear)
    probas4 = linear4 / (1 + sum_linear)
    probas5 = linear5 / (1 + sum_linear)
    probas6 = linear6 / (1 + sum_linear)
    probas7 = linear7 / (1 + sum_linear)
    probas8 = linear8 / (1 + sum_linear)
    probas9 = 1 / (1 + sum_linear)
    # print('Somme des 4 probas : ', probas1+probas2+probas3+probas4)
    array_probas = np.array(
        (
            probas1,
            probas2,
            probas3,
            probas4,
            probas5,
            probas6,
            probas7,
            probas8,
            probas9,
        )
    ).T
    # print('array_probas shape :', array_probas.shape)

    pred_log = np.array(
        [
            np.random.multinomial(n=1, pvals=array_probas[i, :])
            for i in range(len(array_probas))
        ]
    )
    # pred_log = np.random.multinomial(n=1,pvals=array_probas)
    # pred_log = np.apply_along_axis(np.random.multinomial,1,array_probas,{'n':1})
    # print('pred_log shape : ',pred_log.shape)
    # print('pred_log',pred_log)

    array_argmax = np.argmax(pred_log, axis=1)
    # print('array_argmax shape : ',array_argmax.shape)
    # print('array_argmax',array_argmax)

    array_final = np.char.replace(array_argmax.astype(str), "0", list_modalities[0])
    array_final = np.char.replace(array_final.astype(str), "1", list_modalities[1])
    array_final = np.char.replace(array_final.astype(str), "2", list_modalities[2])
    array_final = np.char.replace(array_final.astype(str), "3", list_modalities[3])
    array_final = np.char.replace(array_final.astype(str), "4", list_modalities[4])
    array_final = np.char.replace(array_final.astype(str), "5", list_modalities[5])
    array_final = np.char.replace(array_final.astype(str), "6", list_modalities[6])
    array_final = np.char.replace(array_final.astype(str), "7", list_modalities[7])
    array_final = np.char.replace(array_final.astype(str), "8", list_modalities[8])
    # print('array_final shape ',array_final)

    return array_final, array_argmax


def generate_initial_data_twocat_lgbm5_old(
    dimension, n_samples, random_state=123, verbose=0
):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0, high=1, size=(n_samples, 1))
    for i in range(dimension - 3):
        np.random.seed(seed=random_state + i + 1)
        curr_covariate = np.random.uniform(low=0, high=1, size=(n_samples, 1))
        Xf = np.hstack((Xf, curr_covariate))
    ### Feature categorical

    z_feature_cat_uniform, z_feature_cat_uniform_numeric = (
        generate_synthetic_features_multinomial_nonuple(
            X=Xf,
            index_informatives=[0, 1, 2],
            list_modalities=["Ae", "Bd", "Af", "Ce", "Ad", "Be", "Bf", "Cd", "Cf"],
            beta1=np.array([1, 3, 2]),
            beta2=np.array([4, -7, 3]),
            beta3=np.array([5, -1, 6]),
            beta4=np.array([3, 2, 1]),
            beta5=np.array([2, 5, 1]),
            beta6=np.array([3, 2, 8]),
            beta7=np.array([7, 6, 6]),
            beta8=np.array([1, 3, 1]),
            beta9=np.array([1, 1, 9]),
            intercept=-2,
        )
    )
    # print('z_feature_cat_uniform shape :',z_feature_cat_uniform_numeric.shape)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    print(enc.get_feature_names_out())
    X_final_enc = np.hstack((Xf, X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0, 1, 2]
    list_index_informatives_cat = [-(i + 1) for i in range(n_modalities)]
    print(list_index_informatives_cat)
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    beta = [3, 5.1, 4, -100, 4, 3, 3.2, -100, -100, -100, 5, -100]
    # print('list_index_informatives : ',list_index_informatives)
    # print(beta[:(n_modalities+3)])
    # print(X_final_enc[:,list_index_informatives].shape)

    target, target_numeric = generate_synthetic_features_logreg(
        X=X_final_enc,
        index_informatives=list_index_informatives,
        list_modalities=["No", "Yes"],
        beta=beta,
        treshold=0.5,
        intercept=-11.6,  # intercept=-3
    )
    # print('target_numeric.shape',target_numeric.shape)
    # print('target_numeric', target_numeric)

    first_cat_feature = np.array(
        [z_feature_cat_uniform[i][0] for i in range(n_samples)]
    )
    second_cat_feature = np.array(
        [z_feature_cat_uniform[i][1] for i in range(n_samples)]
    )
    X_final = np.hstack(
        (Xf, first_cat_feature.reshape(-1, 1), second_cat_feature.reshape(-1, 1))
    )

    if verbose == 0:
        print("Composition of the target ", Counter(target_numeric))
        print("Composition of categorical feature : ", Counter(z_feature_cat_uniform))
    return X_final, target, target_numeric


def generate_initial_data_twocat_lgbm5_oldcoeffes(
    dimension, n_samples, random_state=123, verbose=0
):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0, high=1, size=(n_samples, 1))
    for i in range(dimension - 7):
        np.random.seed(seed=random_state + i + 1)
        curr_covariate = np.random.uniform(low=0, high=1, size=(n_samples, 1))
        Xf = np.hstack((Xf, curr_covariate))
    ### Feature categorical

    z_feature_cat_uniform, z_feature_cat_uniform_numeric = (
        generate_synthetic_features_multinomial_nonuple(
            X=Xf,
            index_informatives=[0, 1, 2],
            list_modalities=["Ae", "Bd", "Af", "Ce", "Ad", "Be", "Bf", "Cd", "Cf"],
            beta1=np.array([1, 3, 2]),
            beta2=np.array([4, -7, 3]),
            beta3=np.array([5, -1, 6]),
            beta4=np.array([3, 2, 1]),
            beta5=np.array([2, 5, 1]),
            beta6=np.array([3, 2, 8]),
            beta7=np.array([7, 6, 6]),
            beta8=np.array([1, 3, 1]),
            beta9=np.array([1, 1, 9]),
            intercept=-2,
        )
    )
    # print('z_feature_cat_uniform shape :',z_feature_cat_uniform_numeric.shape)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    print(enc.get_feature_names_out())
    X_final_enc = np.hstack((Xf, X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0, 1, 2]
    list_index_informatives_cat = [-(i + 1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    print("list_index_informatives : ", list_index_informatives)
    beta = [3, -5.1, -4, -8, 4, -3, 3.2, -8, -7, -9, 5, -10]
    # print('list_index_informatives : ',list_index_informatives)
    # print(beta[:(n_modalities+3)])
    # print(X_final_enc[:,list_index_informatives].shape)

    target, target_numeric = generate_synthetic_features_logreg(
        X=X_final_enc,
        index_informatives=list_index_informatives,
        list_modalities=["No", "Yes"],
        beta=beta,
        treshold=0.5,
        intercept=-1.9,  # intercept=-3
    )
    # print('target_numeric.shape',target_numeric.shape)
    # print('target_numeric', target_numeric)

    first_cat_feature = np.array(
        [z_feature_cat_uniform[i][0] for i in range(n_samples)]
    )
    second_cat_feature = np.array(
        [z_feature_cat_uniform[i][1] for i in range(n_samples)]
    )
    X_final = np.hstack(
        (Xf, first_cat_feature.reshape(-1, 1), second_cat_feature.reshape(-1, 1))
    )

    if verbose == 0:
        print("Composition of the target ", Counter(target_numeric))
        print("Composition of categorical feature : ", Counter(z_feature_cat_uniform))
    # return X_final,target,target_numeric
    return X_final, target, target_numeric


def generate_initial_data_twocat_lgbm5_19_11_2024(
    dimension, n_samples, random_state=123, verbose=0
):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0, high=1, size=(n_samples, 1))
    for i in range(dimension - 7):
        np.random.seed(seed=random_state + i + 1)
        curr_covariate = np.random.uniform(low=0, high=1, size=(n_samples, 1))
        Xf = np.hstack((Xf, curr_covariate))
    ### Feature categorical

    z_feature_cat_uniform, z_feature_cat_uniform_numeric = (
        generate_synthetic_features_multinomial_nonuple(
            X=Xf,
            index_informatives=[0, 1, 2],
            list_modalities=["Ae", "Bd", "Af", "Ce", "Ad", "Be", "Bf", "Cd", "Cf"],
            beta1=np.array([1, 3, 2]),
            beta2=np.array([4, -7, 3]),
            beta3=np.array([5, -1, 6]),
            beta4=np.array([3, 2, 1]),
            beta5=np.array([2, 5, 1]),
            beta6=np.array([3, 2, 8]),
            beta7=np.array([7, 6, 6]),
            beta8=np.array([1, 3, 1]),
            beta9=np.array([1, 1, 9]),
            intercept=-2,
        )
    )
    # print('z_feature_cat_uniform shape :',z_feature_cat_uniform_numeric.shape)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    print(enc.get_feature_names_out())
    X_final_enc = np.hstack((Xf, X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0, 1, 2]
    list_index_informatives_cat = [-(i + 1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    print("list_index_informatives : ", list_index_informatives)
    beta = [3, -5.1, -4, -8, -1.2, 1.3, 2.4, -8, -8, -9, -0.5, -10]
    # print('list_index_informatives : ',list_index_informatives)
    # print(beta[:(n_modalities+3)])
    # print(X_final_enc[:,list_index_informatives].shape)

    target, target_numeric = generate_synthetic_features_logreg(
        X=X_final_enc,
        index_informatives=list_index_informatives,
        list_modalities=["No", "Yes"],
        beta=beta,
        treshold=0.5,
        intercept=0.9,  # intercept=-3
    )
    # print('target_numeric.shape',target_numeric.shape)
    # print('target_numeric', target_numeric)

    first_cat_feature = np.array(
        [z_feature_cat_uniform[i][0] for i in range(n_samples)]
    )
    second_cat_feature = np.array(
        [z_feature_cat_uniform[i][1] for i in range(n_samples)]
    )
    X_final = np.hstack(
        (Xf, first_cat_feature.reshape(-1, 1), second_cat_feature.reshape(-1, 1))
    )

    if verbose == 0:
        print("Composition of the target ", Counter(target_numeric))
        print("Composition of categorical feature : ", Counter(z_feature_cat_uniform))
    # return X_final,target,target_numeric
    return X_final, target, target_numeric


def generate_initial_data_twocat_lgbm5(
    dimension_continuous, n_samples, random_state=123, verbose=0
):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0, high=1, size=(n_samples, 1))
    for i in range(dimension_continuous - 1):
        np.random.seed(seed=random_state + i + 1)
        curr_covariate = np.random.uniform(low=0, high=1, size=(n_samples, 1))
        Xf = np.hstack((Xf, curr_covariate))
    ### Feature categorical

    z_feature_cat_uniform, z_feature_cat_uniform_numeric = (
        generate_synthetic_features_multinomial_nonuple(
            X=Xf,
            index_informatives=[0, 1, 2],
            list_modalities=["Ae", "Bd", "Af", "Ce", "Ad", "Be", "Bf", "Cd", "Cf"],
            beta1=np.array([4.1, -1.1, 10.7]),
            beta2=np.array([4.3, -2.1, 11.1]),
            beta3=np.array([4.8, -0.7, 9.7]),
            beta4=np.array([5, -1, 10]),
            beta5=np.array([-5, -1, 10]),
            beta6=np.array([-5, -1, 10]),
            beta7=np.array([4.3, -2.1, 11.1]),
            beta8=np.array([4.3, -2.1, 11.1]),
            beta9=np.array([1, 1, 9]),
            intercept=2,
        )
    )
    # print('z_feature_cat_uniform shape :',z_feature_cat_uniform_numeric.shape)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    print(enc.get_feature_names_out())
    X_final_enc = np.hstack((Xf, X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0, 1, 2]
    list_index_informatives_cat = [-(i + 1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    print("list_index_informatives : ", list_index_informatives)
    beta = [8, -10.1, -9, -11.3, -10.2, 1.3, 1.8, -11.8, -12.2, -10.7, 0.9, -12]
    # print('list_index_informatives : ',list_index_informatives)
    # print(beta[:(n_modalities+3)])
    # print(X_final_enc[:,list_index_informatives].shape)

    target, target_numeric = generate_synthetic_features_logreg(
        X=X_final_enc,
        index_informatives=list_index_informatives,
        list_modalities=["No", "Yes"],
        beta=beta,
        treshold=0.5,
        intercept=0.5,  # intercept=-3
    )

    # print('target_numeric.shape',target_numeric.shape)
    # print('target_numeric', target_numeric)

    first_cat_feature = np.array(
        [z_feature_cat_uniform[i][0] for i in range(n_samples)]
    )
    second_cat_feature = np.array(
        [z_feature_cat_uniform[i][1] for i in range(n_samples)]
    )
    X_final = np.hstack(
        (Xf, first_cat_feature.reshape(-1, 1), second_cat_feature.reshape(-1, 1))
    )

    if verbose == 0:
        print("Composition of the target ", Counter(target_numeric))
        print("Composition of categorical feature : ", Counter(z_feature_cat_uniform))
    # return X_final,target,target_numeric
    return X_final, target, target_numeric
