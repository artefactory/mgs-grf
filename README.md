<div align="center">

# MGS-GRF

</div>

If you face *imbalance data* in your machine learning project, this package is here to pre-process your data. It is an efficient and ready-to-use implementation of
MGS-GRF, an oversampling strategy presented at ECML-PKDD 2025 conference, designed to handle large-scale and mixed imbalanced data-set — with *both continuous and categorical features*.


## 🛠 Installation

First you can clone the repository:
```bash
git clone git@github.com:artefactory/mgs-grf.git
```

## 🚀 How to use the MGS-GRF Algorithm to learn on imbalanced data
Here is a short example on how to use MGS-GRF: 
```python
from mgs_grf import MGSGRFOverSampler

## Apply MGS-GRF procedure to oversample the data
mgs_grf = MGSGRFOverSampler(K=len(numeric_features),categorical_features=categorical_features,random_state=0)
balanced_X, balanced_y = mgs_grf.fit_resample(X_train,y_train)
print("Augmented data : ", Counter(balanced_y))

## Encode the categorical variables
enc = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
balanced_X_encoded = enc.fit_transform(balanced_X[:,categorical_features])
balanced_X_final = np.hstack((balanced_X[:,numeric_features],balanced_X_encoded))

# Fit the final classifier on the augmented data
clf_mgs = lgb.LGBMClassifier(n_estimators=100,verbosity=-1, random_state=0)
clf_mgs.fit(balanced_X_final, balanced_y)

```
A more detailed notebook example is available in [this notebook](example/example.ipynb).


## 🔬 Reproducing the paper experiments

If you want to reproduce our paper experiments:
  - Section 4.2 : the [Python file](protocols/run_synthetic_coherence.py) reproduces the experiments (data sets, oversampling and traing). Then the results can be analyzed with [this notebook](protocols/notebooks/res_coh.ipynb).
  - Section 4.3 : the   [Python file](protocols/run_synthetic_association.py) reproduces the experiments (data sets, oversampling and traing). Then the results can be analyzed with [this notebook](protocols/notebooks/res_asso.ipynb).
  - Section 5 : the [Python file](protocols/run_protocol-final.py) reproduces the experiments (data sets, oversampling and traing). Then the results can be analyzed with [this notebook](protocols/notebooks/res_real_data.ipynb).

## 💾 Data sets

The data sets of used for our article should be dowloaded  inside the *data/externals* folder. The data sets are available at the followings adresses :

* [BankMarketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* [BankChurners](https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m)


## 🙏 Acknowledgements

This work was done through a partenership between **Artefact Research Center** and the **Laboratoire de Probabilités Statistiques et Modélisation** (LPSM) of Sorbonne University.

<p align="center">
  <a href="https://www.artefact.com/data-consulting-transformation/artefact-research-center/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_arc.png" height="80" />
  </a>
  &emsp;
  &emsp;
  <a href="https://www.lpsm.paris/">
    <img src="data/logos//logo_LPSM.jpg" height="95" />
  </a>
</p>


## 📜 Citation

If you find the code useful, please consider citing us :
```
@article{sakho2025harnessing,
  title={Harnessing Mixed Features for Imbalance Data Oversampling: Application to Bank Customers Scoring},
  author={Sakho, Abdoulaye and Malherbe, Emmanuel and Gauthier, Carl-Erik and Scornet, Erwan},
  journal={arXiv preprint arXiv:2503.22730},
  year={2025}
}
```
