<div align="center">

# Do we need rebalancing strategies? A theoretical and empirical study around SMOTE and its variants


Abdoulaye SAKHO<sup>1, 2</sup>, Emmanuel MALHERBE<sup>1</sup>, Erwan SCORNET<sup>2</sup> <br>
 <sup>1</sup> <sub> [Artefact Research Center](https://www.artefact.com/technologies/artefact-research-center/), </sub> <br> <sup>2</sup> <sub>[*LPSM* - Sorbonne Université](https://www.lpsm.paris/),</sub>

In [AISTATS 2026](https://virtual.aistats.org/Conferences/2026). <br>
[[Full Paper]](https://arxiv.org/abs/2402.03819) <br>

</div>

The experiments of this first paper can be found at the following repository: [smote_strategies_study](https://github.com/artefactory/smote_strategies_study)


<div align="center">

# Harnessing Mixed Features for Imbalance Data Oversampling: Application to Bank Customers Scoring


Abdoulaye SAKHO<sup>1, 2</sup>, Emmanuel MALHERBE<sup>1</sup>, Carl-Erik GAUTHIER<sup>3</sup>, Erwan SCORNET<sup>2</sup> <br>
 <sup>1</sup> <sub> [Artefact Research Center](https://www.artefact.com/technologies/artefact-research-center/), </sub> <br> <sup>2</sup> <sub>[*LPSM* - Sorbonne Université](https://www.lpsm.paris/),</sub> <sup>3</sup> <sub>Société Générale</sub>

In [ECML-PKDD 2025](https://ecmlpkdd.org/2025/). <br>
[[Full Paper]](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/ads/preprint_ecml_pkdd_2025_ads_1005.pdf) <br>

</div>


> **Abstract:** *This study investigates rare event detection on tabular data within binary classification. Many real-world classification tasks, such as in banking sector, deal with mixed features, which have a significant impact on predictive performances. To this purpose, we introduce MGS-GRF, an oversampling strategy designed for mixed features. This method uses a kernel density estimator with locally estimated full-rank covariances to generate continuous features, while categorical ones are drawn from the original samples through a generalized random forest.*

You will find here the code to reproduce the paper experiments. If you simply want to test MGS-GRF on your data, you do not need this submodule, please go to the root of the repository for the main documentation.

## 🛠 Installation

Install the **required** packages into your environment (conda, mamba or pip):
```bash
pip install -r requirements.txt
```

## 🔬 Reproducing the paper experiments

If you want to reproduce our paper experiments:
  - Section 4.2 : the [Python file](protocols/run_synthetic_coherence.py) reproduces the experiments (data sets, oversampling and traing). Then the results can be analyzed with [this notebook](protocols/notebooks/res_coh.ipynb).
  - Section 4.3 : the   [Python file](protocols/run_synthetic_association.py) reproduces the experiments (data sets, oversampling and traing). Then the results can be analyzed with [this notebook](protocols/notebooks/res_asso.ipynb).
  - Section 5 : the [Python file](protocols/run_protocol-final.py) reproduces the experiments (data sets, oversampling and traing). Then the results can be analyzed with [this notebook](protocols/notebooks/res_real_data.ipynb).

## 💾 Data sets

The data sets of used for our article should be dowloaded  inside the *data/externals* folder. The data sets are available at the followings adresses :

* [BankMarketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* [BankChurners](https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m)

