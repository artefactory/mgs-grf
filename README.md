## Paper name.
<div align="center">

# Harnessing Mixed Features for Imbalance Data Oversampling: Application to Bank Customers Scoring


Abdoulaye SAKHO<sup>1, 2</sup>, Emmanuel MALHERBE<sup>1</sup>, Carl-Erik GAUTHIER<sup>3</sup>, Erwan SCORNET<sup>2</sup> <br>
 <sup>1</sup> <sub> Artefact Research Center, </sub> <br> <sup>2</sup> <sub>*LPSM* - Sorbonne Université,</sub> <sup>3</sup> <sub>Société Générale</sub>

Preprint. <br>
[[Full Paper]](https://arxiv.org/pdf/2503.22730) <br>

</div>

<p align="center"><img width="95%" src="doc/PLS-3.png" /></p>

**Abstract:** *This study investigates rare event detection on tabular data within binary classification. Standard techniques to handle class imbalance include SMOTE, which generates synthetic samples from the minority class. However, SMOTE is intrinsically designed for continuous input variables. In fact, despite SMOTE-NC—its default extension to handle mixed features (continuous and categorical variables)—very few works propose procedures to synthesize mixed features. On the other hand, many
real-world classification tasks, such as in banking sector, deal with mixed features, which have a significant impact on predictive performances. To this purpose, we introduce MGS-GRF, an oversampling strategy designed for mixed features. This method uses a kernel density estimator with locally estimated full-rank covariances to generate continuous features, while categorical ones are drawn from the original samples through a generalized random forest. Empirically, contrary to SMOTE-NC, we show that MGS-GRF exhibits two important properties: (i) the coherence i.e.
the ability to only generate combinations of categorical features that are already present in the original dataset and (ii) association, i.e. the ability to preserve the dependence between continuous and categorical features. We also evaluate the predictive performances of LightGBM classifiers trained on data sets, augmented with synthetic samples from various strategies. Our comparison is performed on simulated and public realworld data sets, as well as on a private data set from a leading financial institution. We observe that synthetic procedures that have the properties of coherence and association display better predictive performances in terms of various predictive metrics (PR and ROC AUC...), with MGSGRF being the best one. Furthermore, our method exhibits promising results for the private banking application, with development pipeline being compliant with regulatory constraints.*

You will find code to reproduce the paper experiments as well as an nice implementation of our *new* and *efficient* strategy for your projects.
## ⭐ Table of Contents
  - [Getting Started](#getting-started)
  - [Data sets](#data-sets)
  - [Acknowledgements](#acknowledgements)

## ⭐ Getting Started

If you want to reproduce our paper experiments:
  - Section 4.2 : the [pyhton file](protocols/run_synthetic_coherence.py) reproduce the experiments (data sets, oversampling and traing). Then the results can be analyzed [here](protocols/notebooks/res_coh.ipynb).
  - Section 4.3 : the   [pyhton file](protocols/run_synthetic_association.py) reproduce the experiments (data sets, oversampling and traing). Then the results can be analyzed [here](protocols/notebooks/res_asso.ipynb).
  - Section 5 : the [pyhton file](protocols/run_protocol-final.py) reproduce the experiments (data sets, oversampling and traing). Then the results can be analyzed [here](protocols/notebooks/res_real_data.ipynb).

## ⭐ Data sets

The data sets of used for our article should be dowloaded  inside the *data/externals* folder. The data sets are available at the followings adresses :

* [BankMarketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* [BankChurners](https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m)


## ⭐ Acknowledgements

This work was done through a partenership between **Artefact Research Center** and the **Laboratoire de Probabilités Statistiques et Modélisation** (LPSM) of Sorbonne University.

[![Artefact](data/logos/logo_arc.png)](https://www.artefact.com/data-consulting-transformation/artefact-research-center/)  |  [![LPSM]( data/logos//logo_LPSM.jpg)](https://www.lpsm.paris/)
:-------------------------:|:-------------------------:

If you find the code usefull, please consider citing us :
```
@article{sakho2025harnessing,
  title={Harnessing Mixed Features for Imbalance Data Oversampling: Application to Bank Customers Scoring},
  author={Sakho, Abdoulaye and Malherbe, Emmanuel and Gauthier, Carl-Erik and Scornet, Erwan},
  journal={arXiv preprint arXiv:2503.22730},
  year={2025}
}
```