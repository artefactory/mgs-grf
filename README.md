<div align="center">

# MGS-GRF
[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/mgs-grf/blob/main/.pre-commit-config.yaml)


[![cite](https://img.shields.io/badge/Citation-BibTeX-cyan)](./CITATION.bib)


</div>

If you face *imbalance data* in your machine learning project, this package is here to pre-process your data. It is an efficient and ready-to-use implementation of
MGS-GRF, an oversampling strategy presented at [ECML-PKDD 2025](https://ecmlpkdd.org/2025/) conference, designed to handle large-scale and mixed imbalanced data-set — with *both continuous and categorical features*.


## 🛠 Installation

First you can clone the repository:
```bash
git clone git@github.com:artefactory/mgs-grf.git
```

And install the required packages into your environment (conda, mamba or pip):
```bash
pip install -r requirements.txt
```

## 🚀 How to use the MGS-GRF Algorithm to learn on imbalanced data
Here is a short example on how to use MGS-GRF: 
```python
from mgs_grf import MGSGRFOverSampler

## Apply MGS-GRF procedure to oversample the data
mgs_grf = MGSGRFOverSampler(categorical_features=categorical_features, random_state=0)
X_train_balanced, y_train_balanced = mgs_grf.fit_resample(X_train_imbalanced, y_train_imbalanced)

## Encode the categorical variables
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_train_balanced_enc = np.hstack((X_train_balanced[:,numeric_features],
                                  enc.fit_transform(X_train_balanced[:,categorical_features])))
X_test_enc = np.hstack((X_test[:,numeric_features], enc.transform(X_test[:,categorical_features])))

# Fit the final classifier on the augmented data
clf = lgb.LGBMClassifier(n_estimators=100, verbosity=-1, random_state=0)
clf.fit(X_train_balanced_enc, y_train_balanced)

```
A more detailed notebook example is available in [this notebook](example/example.ipynb).


## 🙏 Acknowledgements

This work was done through a partenership between **Artefact Research Center** and the **Laboratoire de Probabilités Statistiques et Modélisation** (LPSM) of Sorbonne University.

<p align="center">
  <a href="https://www.artefact.com/data-consulting-transformation/artefact-research-center/">
    <img src="https://raw.githubusercontent.com/artefactory/choice-learn/main/docs/illustrations/logos/logo_arc.png" height="80" />
  </a>
  &emsp;
  &emsp;
  <a href="https://www.lpsm.paris/">
    <img src="experiments/data/logos//logo_LPSM.jpg" height="95" />
  </a>
</p>


## 📜 Citation

If you find the code useful, please consider citing us :
```
@inproceedings{sakho2025harnessing,
  title={Harnessing Mixed Features for Imbalance Data Oversampling: Application to Bank Customers Scoring},
  author={Sakho, Abdoulaye and Malherbe, Emmanuel and Gauthier, Carl-Erik and Scornet, Erwan},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={247--264},
  year={2025},
  organization={Springer}
}
```
