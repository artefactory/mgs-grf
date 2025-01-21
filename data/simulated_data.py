import numpy as np
from collections import Counter

from sklearn.preprocessing import OneHotEncoder

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
    real_pred_log = np.random.binomial(n=1,p=res_log_reg)
    array_final = np.char.replace(real_pred_log.astype(str), '0',list_modalities[0])
    array_final = np.char.replace(array_final.astype(str), '1',list_modalities[1])
    return array_final,real_pred_log

def my_exp_coeff(x,beta=np.array([-8,7,6]),intercept = -2): 
    tmp = x.dot(beta)
    z = tmp + intercept # add intercept
    res = np.exp(z) 
    return  res

def generate_synthetic_features_multinomial(X,index_informatives,list_modalities,list_beta,list_intercept):
    n_modalities = len(list_modalities)
    list_linear = []

    for i in range(n_modalities):
        res = my_exp_coeff(X[:,index_informatives],beta=list_beta[i],intercept=list_intercept[i])
        list_linear.append(res)
    sum_linear = sum(list_linear)
    list_probas = []
    for i in range(n_modalities):
        res = list_linear[i] / ( sum_linear)
        list_probas.append(res)
    #list_probas.append(1 / (1 + sum_linear)) 

    array_probas = np.array(list_probas).T
    pred_log = np.array([np.random.multinomial(n=1,pvals=array_probas[i,:]) for i in range(len(array_probas))])
    array_argmax = np.argmax(pred_log,axis=1)

    array_final = array_argmax.astype(str)
    for i in range(n_modalities):
        array_final[array_final==str(i)] = list_modalities[i]
        #array_final = np.char.replace(array_final.astype(str), str(i),list_modalities[i])
    return array_final,array_argmax

def generate_initial_data_twocat_case1(dimension_continuous,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension_continuous-1):
        np.random.seed(seed=random_state+i)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
    ### Feature categorical
    beta1 = np.array([11,-5,-6])
    beta0 = np.array([-1000,-1000,-1000])
    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial(
        X=Xf,index_informatives=[0,1,2],
        list_modalities=['Aa','Ab','Ac','Ad','Ae','Af','Ag','Ah','Ai',
                         'Ba','Bb','Bc','Bd','Be','Bf','Bg','Bh','Bi',
                         'Ca','Cb','Cc','Cd','Ce','Cf','Cg','Ch','Ci',
                         'Da','Db','Dc','Dd','De','Df','Dg','Dh','Di',
                         'Ea','Eb','Ec','Ed','Ee','Ef','Eg','Eh','Ei',
                         'Fa','Fb','Fc','Fd','Fe','Ff','Fg','Fh','Fi',
                         'Ga','Gb','Gc','Gd','Ge','Gf','Gg','Gh','Gi',
                         'Ha','Hb','Hc','Hd','He','Hf','Hg','Hh','Hi',
                         'Ia','Ib','Ic','Id','Ie','If','Ig','Ih','Ii',],
        list_beta = [beta1,beta1,beta1,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta1,beta1,beta1,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta1,beta1,beta1,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,],
        list_intercept= [2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                        ])
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    #beta = [8,-10.1,-9,-11.3,-12.2,1.3,1.8,-11.8,-12.2,-10.7,0.9,-12,-12,-12,-12,-12,-12,-12,-12,-12,
    beta = [8,-10.1,-9,
            3.0,-12,-12,
            -12,3.0,-12,
            -12,-12,3.0,
           ]

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=0.5
                                                               )
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    #return X_final,target,target_numeric
    return X_final,target,target_numeric

def generate_initial_data_twocat_case2(dimension_continuous,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension_continuous-1):
        np.random.seed(seed=random_state+i)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
    ### Feature categorical
    beta1 = np.array([11,-5,-6])
    beta0 = np.array([-1000,-1000,-1000])
    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial(
        X=Xf,index_informatives=[0,1,2],
        list_modalities=['Aa','Ab','Ac','Ad','Ae','Af','Ag','Ah','Ai',
                         'Ba','Bb','Bc','Bd','Be','Bf','Bg','Bh','Bi',
                         'Ca','Cb','Cc','Cd','Ce','Cf','Cg','Ch','Ci',
                         'Da','Db','Dc','Dd','De','Df','Dg','Dh','Di',
                         'Ea','Eb','Ec','Ed','Ee','Ef','Eg','Eh','Ei',
                         'Fa','Fb','Fc','Fd','Fe','Ff','Fg','Fh','Fi',
                         'Ga','Gb','Gc','Gd','Ge','Gf','Gg','Gh','Gi',
                         'Ha','Hb','Hc','Hd','He','Hf','Hg','Hh','Hi',
                         'Ia','Ib','Ic','Id','Ie','If','Ig','Ih','Ii',],
        list_beta = [beta1,beta1,beta1,beta1,beta1,beta0,beta0,beta0,beta0,
                     beta1,beta1,beta1,beta1,beta1,beta0,beta0,beta0,beta0,
                     beta1,beta1,beta1,beta1,beta1,beta0,beta0,beta0,beta0,
                     beta1,beta1,beta1,beta1,beta1,beta0,beta0,beta0,beta0,
                     beta1,beta1,beta1,beta1,beta1,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,],
        list_intercept= [2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                        ])
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,X_final_cat_enc))
    
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    beta = [8,-10.1,-9,
            3.0,-12,-12,-12,-12,
            -12,3.0,-12,-12,-12,
            -12,-12,3.0,-12,-12,
            -12,-12,-12,3.0,-12,
            -12,-12,-12,-12,3.0,
           ]

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=1.0# intercept=-3
                                                               )
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    return X_final,target,target_numeric

def generate_initial_data_twocat_case3(dimension_continuous,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension_continuous-1):
        np.random.seed(seed=random_state+i)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
    ### Feature categorical
    beta1 = np.array([11,-5,-6])
    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial(
        X=Xf,index_informatives=[0,1,2],
        list_modalities=['Aa','Ab','Ac','Ad','Ae','Af','Ag','Ah','Ai',
                         'Ba','Bb','Bc','Bd','Be','Bf','Bg','Bh','Bi',
                         'Ca','Cb','Cc','Cd','Ce','Cf','Cg','Ch','Ci',
                         'Da','Db','Dc','Dd','De','Df','Dg','Dh','Di',
                         'Ea','Eb','Ec','Ed','Ee','Ef','Eg','Eh','Ei',
                         'Fa','Fb','Fc','Fd','Fe','Ff','Fg','Fh','Fi',
                         'Ga','Gb','Gc','Gd','Ge','Gf','Gg','Gh','Gi',
                         'Ha','Hb','Hc','Hd','He','Hf','Hg','Hh','Hi',
                         'Ia','Ib','Ic','Id','Ie','If','Ig','Ih','Ii',],
        list_beta = [beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,],
        list_intercept= [2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,
                        ])
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,X_final_cat_enc))
    
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    beta = [8,-10.1,-9,
            3.0,-12,-12,-12,-12,-12,-12,-12,-12,
            -12,3.0,-12,-12,-12,-12,-12,-12,-12,
            -12,-12,3.0,-12,-12,-12,-12,-12,-12,
            -12,-12,-12,3.0,-12,-12,-12,-12,-12,
            -12,-12,-12,-12,3.0,-12,-12,-12,-12,
            -12,-12,-12,-12,-12,3.0,-12,-12,-12,
            -12,-12,-12,-12,-12,-12,3.0,-12,-12,
            -12,-12,-12,-12,-12,-12,-12,3.0,-12,
            -12,-12,-12,-12,-12,-12,-12,-12,3.0,
           ]

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=3.0# intercept=-3
                                                               )
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    return X_final,target,target_numeric


def generate_initial_data_twocat_case4(dimension_continuous,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension_continuous-1):
        np.random.seed(seed=random_state+i)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
    ### Feature categorical
    beta1 = np.array([11,-5,-6])
    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial(
        X=Xf,index_informatives=[0,1,2],
        list_modalities=['Aa','Ab','Ac','Ad','Ae','Af','Ag','Ah','Ai','Aj','Ak','Al','Am','An','Ao',
                         'Ba','Bb','Bc','Bd','Be','Bf','Bg','Bh','Bi','Bj','Bk','Bl','Bm','Bn','Bo',
                         'Ca','Cb','Cc','Cd','Ce','Cf','Cg','Ch','Ci','Cj','Ck','Cl','Cm','Cn','Co',
                         'Da','Db','Dc','Dd','De','Df','Dg','Dh','Di','Dj','Dk','Dl','Dm','Dn','Do',
                         'Ea','Eb','Ec','Ed','Ee','Ef','Eg','Eh','Ei','Ej','Ek','El','Em','En','Eo',
                         'Fa','Fb','Fc','Fd','Fe','Ff','Fg','Fh','Fi','Fj','Fk','Fl','Fm','Fn','Fo',
                         'Ga','Gb','Gc','Gd','Ge','Gf','Gg','Gh','Gi','Gj','Gk','Gl','Gm','Gn','Go',
                         'Ha','Hb','Hc','Hd','He','Hf','Hg','Hh','Hi','Hj','Hk','Hl','Hm','Hn','Ho',
                         'Ia','Ib','Ic','Id','Ie','If','Ig','Ih','Ii','Ij','Ik','Il','Im','In','Io',
                         'Ja','Jb','Jc','Jd','Je','Jf','Jg','Jh','Ji','Jj','Jk','Jl','Jm','Jn','Jo',
                         'Ka','Kb','Kc','Kd','Ke','Kf','Kg','Kh','Ki','Kj','Kk','Kl','Km','Kn','Ko',
                         'La','Lb','Lc','Ld','Le','Lf','Lg','Lh','Li','Lj','Lk','Ll','Lm','Ln','Lo',
                         'Ma','Mb','Mc','Md','Me','Mf','Mg','Mh','Mi','Mj','Mk','Ml','Mm','Mn','Mo',
                         'Na','Nb','Nc','Nd','Ne','Nf','Ng','Nh','Ni','Nj','Nk','Nl','Nm','Nn','No',
                         'Oa','Ob','Oc','Od','Oe','Of','Og','Oh','Oi','Oj','Ok','Ol','Om','On','Oo',
                        ],
        list_beta = [beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                     beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,beta1,
                    ],
        list_intercept= [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                         2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                        ])
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,X_final_cat_enc))
    
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    #beta = [8,-10.1,-9,-11.3,-12.2,1.3,1.8,-11.8,-12.2,-10.7,0.9,-12,-12,-12,-12,-12,-12,-12,-12,-12,
    beta = [8,-10.1,-9,
            3.0,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,
            -12,3.0,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,
            -12,-12,3.0,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,
            -12,-12,-12,3.0,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,
            -12,-12,-12,-12,3.0,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,
            -12,-12,-12,-12,-12,3.0,-12,-12,-12,-12,-12,-12,-12,-12,-12,
            -12,-12,-12,-12,-12,-12,3.0,-12,-12,-12,-12,-12,-12,-12,-12,
            -12,-12,-12,-12,-12,-12,-12,3.0,-12,-12,-12,-12,-12,-12,-12,
            -12,-12,-12,-12,-12,-12,-12,-12,3.0,-12,-12,-12,-12,-12,-12,
            -12,-12,-12,-12,-12,-12,-12,-12,-12,3.0,-12,-12,-12,-12,-12,
            -12,-12,-12,-12,-12,-12,-12,-12,-12,-12,3.0,-12,-12,-12,-12,
            -12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,3.0,-12,-12,-12,
            -12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,3.0,-12,-12,
            -12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,3.0,-12,
            -12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,3.0,
           ]
    print('len  beta',len(beta))
    print('len list_index_informatives',len(list_index_informatives))

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=2.5# intercept=-1.5
                                                               )
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    return X_final,target,target_numeric

def generate_initial_data_twocat_case5(dimension_continuous,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.uniform(low=0,high=1,size=(n_samples,1))
    for i in range(dimension_continuous-1):
        np.random.seed(seed=random_state+i)
        curr_covariate = np.random.uniform(low=0,high=1,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
    ### Feature categorical
    
    majuscules = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U']
    minuscules = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u']
    list_modalities=[ X+x for X in majuscules for x in minuscules]
    list_beta = [[11,-5,-6] for _ in range(len(list_modalities))]

    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial(
        X=Xf,index_informatives=[0,1,2],
        list_modalities=list_modalities,
        list_beta = list_beta,
        list_intercept= [2 for _ in range(len(list_modalities))])
    
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,X_final_cat_enc))
    
    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    beta = [8,-10.1,-9]
    beta.extend( 
        (np.full((len(majuscules),len(majuscules)),-12) + np.diag(np.full((len(majuscules)),12+3)) ).ravel().tolist() )

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=4.5# intercept=-3
                                                               )
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    return X_final,target,target_numeric