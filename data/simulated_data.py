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

########################################################################
################ Normal distribution simulations 2 #####################
########################################################################

def generate_initial_data_twocat_normal_case1(n_samples,mean,cov,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf=np.random.multivariate_normal(mean=mean,cov=cov,size=n_samples)
    ### Feature categorical
    beta0 = np.array([0,0,0])
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
        list_beta = [np.array([11,-5,-6]),np.array([10.7,-4.8,-6.1]),np.array([11.3,-5,-6.3]),beta0,beta0,beta0,beta0,beta0,beta0,
                     np.array([11.1,-5.1,-6.1]),np.array([11,-5,-6]),np.array([11,-5,-6]),beta0,beta0,beta0,beta0,beta0,beta0,
                     np.array([10.9,-5.3,-6.3]),np.array([11.2,-5.2,-5.7]),np.array([11,-4.7,-6.1]),beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,],
        list_intercept= [2,2,2,-200,-200,-200,-200,-200,-200,
                         2,2,2,-200,-200,-200,-200,-200,-200,
                         2,2,2,-200,-200,-200,-200,-200,-200,
                         -200,-200,-200,-200,-200,-200,-200,-200,-200,
                         -200,-200,-200,-200,-200,-200,-200,-200,-200,
                         -200,-200,-200,-200,-200,-200,-200,-200,-200,
                         -200,-200,-200,-200,-200,-200,-200,-200,-200,
                         -200,-200,-200,-200,-200,-200,-200,-200,-200,
                         -200,-200,-200,-200,-200,-200,-200,-200,-200,
                        ])
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    beta = [8,-10.1,-2,
            30.0,-32,-32,
            -32,30.0,-32,
            -32,-32,30.0,
           ]

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=-38
                                                               )
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        
    return X_final,target,target_numeric

def generate_initial_data_twocat_normal_case2(n_samples,mean,cov,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf=np.random.multivariate_normal(mean=mean,cov=cov,size=n_samples)
    ### Feature categorical
    beta0 = np.array([0,0,0])
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
        list_beta = [np.array([11,-5,-6]),np.array([10.7,-4.8,-6.1]),np.array([11.3,-5,-6.3]),np.array([9,-5,-6]),np.array([10.1,-4,-5]),beta0,beta0,beta0,beta0,
                     np.array([11.1,-5.1,-6.1]),np.array([11,-5,-6]),np.array([11,-5,-6]),np.array([11,-4.5,-6.5]),np.array([11.3,-4.1,-6]),beta0,beta0,beta0,beta0,
                     np.array([10.9,-5.3,-6.3]),np.array([11.2,-5.2,-5.7]),np.array([11,-4.7,-6.1]),np.array([10.3,-5,-6]),np.array([10.1,-5,-6]),beta0,beta0,beta0,beta0,
                     np.array([10.5,-4.8,-6.6]),np.array([10,-4,-6]),np.array([11,-4.7,-6]),np.array([10.6,-5,-6]),np.array([11,-4,-6]),beta0,beta0,beta0,beta0,
                     np.array([11.4,-5.5,-6]),np.array([11,-4.8,-6]),np.array([11,-5.6,-6]),np.array([11,-5,-6]),np.array([10.6,-5,-6]),beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                     beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,beta0,
                    ],
        list_intercept= [2,2,2,2,2,-200,-200,-200,-200,
                         2,2,2,2,2,-200,-200,-200,-200,
                         2,2,2,2,2,-200,-200,-200,-200,
                         2,2,2,2,2,-200,-200,-200,-200,
                         2,2,2,2,2,-200,-200,-200,-200,
                         -200,-200,-200,-200,-200,-200,-200,-200,-200,
                         -200,-200,-200,-200,-200,-200,-200,-200,-200,
                         -200,-200,-200,-200,-200,-200,-200,-200,-200,
                         -200,-200,-200,-200,-200,-200,-200,-200,-200,
                        ])
    print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    beta = [8,-10.1,-2,
            30.0,-32,-32,-32,-32,
            -32,30.0,-32,-32,-32,
            -32,-32,30.0,-32,-32,
            -32,-32,-32,30.0,-32,
            -32,-32,-32,-32,30.0,
           ]

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=-28
                                                               )
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        
    return X_final,target,target_numeric

def generate_initial_data_twocat_normal_case3(n_samples,mean,cov,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf=np.random.multivariate_normal(mean=mean,cov=cov,size=n_samples)
    ### Feature categorical
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
        list_beta = [np.array([11,-5,-6]),np.array([10.7,-4.8,-6.1]),np.array([11.3,-5,-6.3]),np.array([9,-5,-6]),np.array([10.1,-4,-5]),np.array([11,-5,-6]),np.array([10,-5,-6]),np.array([11,-5,-5]),np.array([11,-5,-5]),
                     np.array([11.1,-5.1,-6.1]),np.array([11,-5,-6]),np.array([11,-5,-6]),np.array([11,-4.5,-6.5]),np.array([11.3,-4.1,-6]),np.array([10,-5,-5]),np.array([11,-5,-5]),np.array([11.5,-5,-6]),np.array([11.1,-5,-6]),
                     np.array([10.9,-5.3,-6.3]),np.array([11.2,-5.2,-5.7]),np.array([11,-4.7,-6.1]),np.array([10.3,-5,-6]),np.array([10.1,-5,-6]),np.array([10.12,-5.1,-6]),np.array([11.12,-5.1,-6]),np.array([11,-5.1,-6.5]),np.array([11,-5,-6.3]),
                     np.array([10.5,-4.8,-6.6]),np.array([10,-4,-6]),np.array([11,-4.7,-6]),np.array([10.6,-5,-6]),np.array([11,-4,-6]),np.array([10.8,-5,-6]),np.array([11,-4.6,-6]),np.array([10,-5,-6]),np.array([10.4,-5,-6]),
                     np.array([11.4,-5.5,-6]),np.array([11,-4.8,-6]),np.array([11,-5.6,-6]),np.array([11,-5,-6]),np.array([10.6,-5,-6]),np.array([11,-4.7,-5.3]),np.array([11,-5,-6]),np.array([10.1,-5,-6]),np.array([11,-5,-6]),
                     np.array([10.3,-4.4,-5.5]),np.array([10.2,-5.1,-6]),np.array([10.6,-5.3,-6]),np.array([10.4,-5,-6]),np.array([10.3,-5.6,-6]),np.array([11,-4.8,-6]),np.array([10.9,-5,-6]),np.array([11,-5,-6]),np.array([11,-5,-5.9]),
                     np.array([10.3,-5.3,-6.3]),np.array([10.2,-5,-6.2]),np.array([10.7,-5,-6]),np.array([10.7,-5,-6]),np.array([11,-4.8,-6.3]),np.array([11,-5,-5.8]),np.array([11.3,-5,-6]),np.array([11.4,-4.9,-6]),np.array([11,-5,-5.9]),
                     np.array([11,-5,-5]),np.array([10.1,-5.2,-6.3]),np.array([11,-4.7,-6]),np.array([11,-5,-6]),np.array([11,-5,-6]),np.array([11,-5,-6]),np.array([11,-5,-6]),np.array([11,-5,-6]),np.array([11,-5,-6]),
                     np.array([10,-5,-5]),np.array([11.3,-5.2,-6.1]),np.array([11,-5,-5.7]),np.array([11,-4.9,-5.6]),np.array([10.9,-4.9,-6.1]),np.array([10,-5.2,-6]),np.array([10,-4.9,-6]),np.array([11,-4.2,-5]),np.array([10.9,-5,-6]),],
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

    print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    beta = [8,-10.1,-2,
            30.0,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,30.0,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,30.0,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,30.0,-32,-12,-32,-32,-32,
            -32,-32,-32,-32,30.0,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,30.0,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,30.0,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,30.0,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,30.0,
           ]

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=-15
                                                               )
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        
    return X_final,target,target_numeric

def generate_initial_data_twocat_normal_case4(n_samples,mean,cov,random_state=123,verbose=0):
    np.random.seed(11)
    list_beta = [
        np.array([np.random.uniform(low=10.8,high=11.2),np.random.uniform(low=-6.2,high=-5.8),np.random.uniform(low=-6.2,high=-5.8)])
        for i in range(144)
    ]
    np.random.seed(random_state)
    Xf=np.random.multivariate_normal(mean=mean,cov=cov,size=n_samples)
    ### Feature categorical
    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial(
        X=Xf,index_informatives=[0,1,2],
        list_modalities=['Aa','Ab','Ac','Ad','Ae','Af','Ag','Ah','Ai','Aj','Ak','Al',
                         'Ba','Bb','Bc','Bd','Be','Bf','Bg','Bh','Bi','Bj','Bk','Bl',
                         'Ca','Cb','Cc','Cd','Ce','Cf','Cg','Ch','Ci','Cj','Ck','Cl',
                         'Da','Db','Dc','Dd','De','Df','Dg','Dh','Di','Dj','Dk','Dl',
                         'Ea','Eb','Ec','Ed','Ee','Ef','Eg','Eh','Ei','Ej','Ek','El',
                         'Fa','Fb','Fc','Fd','Fe','Ff','Fg','Fh','Fi','Fj','Fk','Fl',
                         'Ga','Gb','Gc','Gd','Ge','Gf','Gg','Gh','Gi','Gj','Gk','Gl',
                         'Ha','Hb','Hc','Hd','He','Hf','Hg','Hh','Hi','Hj','Hk','Hl',
                         'Ia','Ib','Ic','Id','Ie','If','Ig','Ih','Ii','Ij','Ik','Il',
                         'Ja','Jb','Jc','Jd','Je','Jf','Jg','Jh','Ji','Jj','Jk','Jl',
                         'Ka','Kb','Kc','Kd','Ke','Kf','Kg','Kh','Ki','Kj','Kk','Kl',
                         'La','Lb','Lc','Ld','Le','Lf','Lg','Lh','Li','Lj','Lk','Ll',
        ],
        list_beta = list_beta ,
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
    print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    beta = [8,-10.1,-2,
            30.0,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,30.0,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,30.0,-32,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,30.0,-32,-12,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,30.0,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,30.0,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,30.0,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,30.0,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,30.0,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,-32,30.0,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,-32,-32,30.0,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,30.0,
           ]

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=-18
                                                               )
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
    return X_final,target,target_numeric


def generate_initial_data_twocat_normal_case5(n_samples,mean,cov,random_state=123,verbose=0):
    np.random.seed(11)
    list_beta = [
        np.array([np.random.uniform(low=10.8,high=11.2),np.random.uniform(low=-6.2,high=-5.8),np.random.uniform(low=-6.2,high=-5.8)])
        for i in range(225) # n_modalities
    ]
    np.random.seed(random_state)
    Xf=np.random.multivariate_normal(mean=mean,cov=cov,size=n_samples)
    ### Feature categorical
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
        list_beta = list_beta ,
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
    print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_final_cat_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,X_final_cat_enc))

    n_modalities = len(enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    beta = [8,-10.1,-2,
            30.0,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,30.0,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,30.0,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,30.0,-32,-12,-32,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,30.0,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,30.0,-32,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,30.0,-32,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,30.0,-32,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,30.0,-32,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,-32,30.0,-32,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,-32,-32,30.0,-32,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,30.0,-32,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,30.0,-32,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,30.0,-32,
            -32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,-32,30.0,
           ]

    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=-18
                                                               )
    
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        
    #return X_final,target,target_numeric
    return X_final,target,target_numeric

def generate_initial_data_twocat_normal_case6(n_samples,mean,cov,random_state=123,verbose=0):
    np.random.seed(11)
    majuscules = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U']
    minuscules = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u']
    list_modalities=[ X+x for X in majuscules for x in minuscules]
    list_beta = [
        np.array([np.random.uniform(low=10.8,high=11.2),np.random.uniform(low=-6.2,high=-5.8),np.random.uniform(low=-6.2,high=-5.8)])
        for i in range(len(list_modalities)) # n_modalities
    ]
    np.random.seed(random_state)
    Xf=np.random.multivariate_normal(mean=mean,cov=cov,size=n_samples)
    ### Feature categorical
    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial(
        X=Xf,index_informatives=[0,1,2],
        list_modalities=list_modalities,
        list_beta = list_beta,
        list_intercept= [10 for _ in range(len(list_modalities))])
    
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
        (np.full((len(majuscules),len(majuscules)),-52) + np.diag(np.full((len(majuscules)),52+12)) ).ravel().tolist() )
    
    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=-10# intercept=-3
                                                               )
    first_cat_feature = np.array([z_feature_cat_uniform[i][0] for i in range(n_samples) ])
    second_cat_feature = np.array([z_feature_cat_uniform[i][1] for i in range(n_samples) ])
    X_final = np.hstack((Xf,first_cat_feature.reshape(-1,1),second_cat_feature.reshape(-1,1)))
    
    if verbose==0:
        print('Composition of the target ', Counter(target_numeric))
        print('Composition of categorical feature : ', Counter(z_feature_cat_uniform))
    return X_final,target,target_numeric

##########################################################################################
##########################################################################################
####################################         SIMULATIONS 1 ##############################
##########################################################################################
##########################################################################################

def generate_initial_data_onecat_v2_2025_02_11(dimension_continuous,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    Xf = np.random.normal(loc=2,scale=3,size=(n_samples,1))
    for i in range(dimension_continuous-4):
        np.random.seed(seed=random_state+20+i)
        curr_covariate = np.random.normal(loc=2,scale=3,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
        
    z_p1 = np.eye(3) 
    #z_p1[2,2]=0.1
    z_p2 = np.eye(3) 
    #z_p2[0,0]=0.1
    z_p3 = np.eye(3) 
    #z_p3[1,1]=0.1
    mu_p1 = np.array([4,4,6])
    mu_p2 = np.array([1,1,1])
    mu_p3 = np.array([7,7,7])
    
    X_gmm,z_plan = gmm_sampling(n_samples=n_samples,z=[24/50,24/50,2/50],
                         mus= [mu_p1,mu_p2,mu_p3],covs =[z_p1,z_p2,z_p2] )
    Xf = np.hstack((X_gmm,Xf))
    ### Feature categorical 
    indices = np.arange(0,n_samples,1,dtype=int)
    s1 = indices[z_plan==0]
    s2= indices[z_plan==1]
    s3= indices[z_plan==2]
    Xf_plan1 = Xf[s1,:]
    Xf_plan2 = Xf[s2,:]
    Xf_plan3 = Xf[s3,:]

    ## PLAN 1 :
    print("#####")
    #print('Xf_plan1 : ', Xf_plan1[:100,:3])
    z_feature_cat_uniform_plan1, z_feature_cat_uniform_numeric_plan1 = generate_synthetic_features_logreg(
        X=Xf_plan1,index_informatives=[0,1],list_modalities=['BC','A'],beta=np.array([-2,-2]),treshold=.5,intercept=10.2
    ) 
    print('z_feature_cat_uniform_plan1 : ', Counter(z_feature_cat_uniform_plan1))
    n_maj_plan1 = len(z_feature_cat_uniform_plan1[z_feature_cat_uniform_plan1=='BC'])
    z_feature_cat_uniform_plan1[z_feature_cat_uniform_plan1=='BC'] = np.random.choice(['B','C'], n_maj_plan1, replace=True)
    target_numeric_plan1 = np.zeros((len(z_feature_cat_uniform_plan1),),dtype=int)
    target_numeric_plan1[z_feature_cat_uniform_plan1=='A'] = 1
    print('target_numeric_plan1 : ', Counter(target_numeric_plan1))
    print("#####")

    ## PLAN 2 : 
    z_feature_cat_uniform_plan2, z_feature_cat_uniform_numeric_plan2 = generate_synthetic_features_logreg(
        X=Xf_plan2,index_informatives=[1,2],list_modalities=['AC','B'],beta=np.array([-1.5,-3]),treshold=.5,intercept=-1.8
    ) 
    print('z_feature_cat_uniform_plan2 : ', Counter(z_feature_cat_uniform_plan2))
    n_maj_plan2 = len(z_feature_cat_uniform_plan2[z_feature_cat_uniform_plan2=='AC'])
    z_feature_cat_uniform_plan2[z_feature_cat_uniform_plan2=='AC'] = np.random.choice(['A','C'], n_maj_plan2, replace=True)
    target_numeric_plan2 = np.zeros((len(z_feature_cat_uniform_plan2),),dtype=int)
    target_numeric_plan2[z_feature_cat_uniform_plan2=='B'] = 1
    print('target_numeric_plan2 : ', Counter(target_numeric_plan2))
    print("#####")
    
    ## PLAN 3 :
    z_feature_cat_uniform_plan3, z_feature_cat_uniform_numeric_plan3 = generate_synthetic_features_logreg(
        X=Xf_plan3,index_informatives=[0,2],list_modalities=['AB','C'],beta=np.array([-1,-1]),treshold=.5,intercept=13.9
    ) 
    print('z_feature_cat_uniform_plan3 : ', Counter(z_feature_cat_uniform_plan3))
    n_maj_plan3 = len(z_feature_cat_uniform_plan3[z_feature_cat_uniform_plan3=='AB'])
    z_feature_cat_uniform_plan3[z_feature_cat_uniform_plan3=='AB'] = np.random.choice(['A','B'], n_maj_plan3, replace=True)
    target_numeric_plan3 = np.zeros((len(z_feature_cat_uniform_plan3),),dtype=int)
    target_numeric_plan3[z_feature_cat_uniform_plan3=='C'] = 1
    print('target_numeric_plan3 : ', Counter(target_numeric_plan3))
    print("#####")

    ## final     
    X_final_plan1 = np.hstack((Xf_plan1,z_feature_cat_uniform_plan1.reshape(-1,1)))
    X_final_plan2 = np.hstack((Xf_plan2,z_feature_cat_uniform_plan2.reshape(-1,1)))
    X_final_plan3 = np.hstack((Xf_plan3,z_feature_cat_uniform_plan3.reshape(-1,1)))
    X_final = np.vstack((X_final_plan1,X_final_plan2,X_final_plan3))
    
    target_numeric = np.hstack((target_numeric_plan1,target_numeric_plan2,target_numeric_plan3))
    target_numeric = target_numeric.astype(int)
    print('target_numeric : ', target_numeric)
    
    if verbose==0:
        print("*************"*8)
        print('Composition of the target ', Counter(target_numeric))
    #return X_final,target,target_numeric
    return X_final,target_numeric

def gmm_sampling(n_samples,z,mus,covs):
    components = np.random.choice(list(range(len(z))),size=n_samples,replace=True,p=z)
    list_sample = []
    for i in range(n_samples):
        current_component = components[i]
        sample = np.random.multivariate_normal(mus[current_component], covs[current_component])
        list_sample.append(sample)
    return np.array(list_sample),components

def generate_initial_data_onecat_2025_02_25(dimension_continuous,n_samples,random_state=123,verbose=0):
    np.random.seed(random_state)
    #Xf=np.random.multivariate_normal(mean=2*np.ones(dimension_continuous-3),cov=3*np.eye(dimension_continuous-3),size=n_samples)
    Xf = np.random.normal(loc=2,scale=3,size=(n_samples,1))
    for i in range(dimension_continuous-4):
        np.random.seed(seed=random_state+20+i)
        curr_covariate = np.random.normal(loc=2,scale=3,size=(n_samples,1))
        Xf = np.hstack((Xf,curr_covariate))
        
    z_p1 = np.eye(3) 
    #z_p1[2,2]=0.1
    z_p2 = np.eye(3) 
    #z_p2[0,0]=0.1
    z_p3 = np.eye(3) 
    #z_p3[1,1]=0.1
    mu_p1 = np.array([1,1,1])
    mu_p2 = np.array([4,4,4])
    mu_p3 = np.array([7,7,7])
    
    X_gmm,z_plan = gmm_sampling(n_samples=n_samples,z=[22/50,22/50,6/50],
                         mus= [mu_p1,mu_p2,mu_p3],covs =[z_p1,z_p2,z_p2] )
    Xf = np.hstack((X_gmm,Xf))
    ### Feature categorical 

    z_feature_cat_uniform, z_feature_cat_uniform_numeric = generate_synthetic_features_multinomial(
        X=Xf,index_informatives=[0,1,2],
        list_modalities=['A','B','C'],
        list_beta = [np.array([5,2,-3]),np.array([4.8,2.2,-3.1]),np.array([4.9,1.8,-2.8])],
        list_intercept= [0,0,0,])
    #print('z_feature_cat_uniform : ',Counter(z_feature_cat_uniform))
    
    
    z_feature_cat_uniform = np.array(z_feature_cat_uniform).astype(str)
    z_plan= np.array(z_plan).astype(str)
    z_plan[z_plan==str(0)] = 'a'
    z_plan[z_plan==str(1)] = 'b'
    z_plan[z_plan==str(2)] = 'c'
    #print(z_feature_cat_uniform)
    #print(z_plan)
    zw =  z_feature_cat_uniform.astype(object) + z_plan.astype(object)
    #print('zw : ', zw)
    
    
    enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    zw_enc  = enc.fit_transform(zw.reshape(-1, 1))
    #z_enc = enc.fit_transform(z_feature_cat_uniform.reshape(-1, 1))
    #enc_w = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    #w_enc = enc.fit_transform(z_plan.reshape(-1, 1))
    X_final_enc = np.hstack((Xf,zw_enc))

    n_modalities = len(enc.get_feature_names_out()) 
    #print('enc : ', enc.get_feature_names_out())
    list_index_informatives = [0,1,2]
    list_index_informatives_cat = [-(i+1) for i in range(n_modalities)]
    list_index_informatives_cat.reverse()
    list_index_informatives.extend(list_index_informatives_cat)
    #print('list_index_informatives : ',list_index_informatives)
    beta = [6,-3.1,-5,
            20,-30,-30,
            -30,29,-30,
            -30,-30,50,
           ]    
    target,target_numeric = generate_synthetic_features_logreg(X=X_final_enc,index_informatives=list_index_informatives,list_modalities=['No','Yes'],
                                                beta=beta,treshold=.5,intercept=-28
                                                               )
    X_final = np.hstack((Xf,z_feature_cat_uniform.reshape(-1,1)))
    #print('target_numeric : ', target_numeric)

    #print('Plan a : ',Counter(z_feature_cat_uniform[z_plan=="a"]) , Counter(target_numeric[z_plan=="a"]))
    #print('Plan b : ',Counter(z_feature_cat_uniform[z_plan=="b"]), Counter(target_numeric[z_plan=="b"]) )
    #print('Plan c : ',Counter(z_feature_cat_uniform[z_plan=="c"]), Counter(target_numeric[z_plan=="c"]) )
    print("****")
    print('Plan a : ',Counter(z_feature_cat_uniform[(z_plan=="a") & (target_numeric==1)]))
    print('Plan b : ',Counter(z_feature_cat_uniform[(z_plan=="b") & (target_numeric==1)]))
    print('Plan c : ',Counter(z_feature_cat_uniform[(z_plan=="c") & (target_numeric==1)]))

    if verbose==0:
        print("*************"*8)
        print('Composition of the target ', Counter(target_numeric))
    #return X_final,target,target_numeric
    return X_final,target_numeric