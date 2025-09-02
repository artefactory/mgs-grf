"""

Generalized Random Forest implementation based on scikit-learn, for categorical variables.

"""

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)


def iterative_random_choice(probas):
    """
    Function for applying a np.random.choice several times with succesive values of probas.

    Parameters
    ----------
        probas : np.array like
            Probabilities of the samples.
    """
    thresholds = np.random.uniform(size=len(probas))
    cumulative_weights = np.cumsum(probas, axis=1)
    return np.argmax((cumulative_weights.T > thresholds), axis=0)


class DrfFitPredictMixin:
    """
    Mixin for the Genralized Random Forest Procedure.
    The predict draw a sample based on the frequency of training samples ending in the same leaf as the new sample.
    """

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values (class labels in classification, real numbers in regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        super().fit(X=X, y=y, sample_weight=sample_weight)
        self.train_y = y
        self.train_samples_leaves = (
            super().apply(X).astype(np.int32)
        )  # train_samples_leaves: size n_train x n_trees

    def get_weights(self, X):
        """
        Derive frequency of training samples ending in the same leaf as the new sample X.
        (see GRF algorithm for details)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New samples for which to compute the weights.

        Returns
        -------
        np.ndarray of shape (n_samples, n_train)
            Weights for each sample in X based on the training samples leaves.
            Each element is the frequency of the training sample's leaf in the new sample.
        """
        leafs_by_sample = super().apply(X).astype(np.int32)  # taille n_samples x n_trees
        leaves_match = np.array(
            [leafs_by_sample[i] == self.train_samples_leaves for i in range(len(X))]
        )
        n_by_tree = leaves_match.sum(axis=1)[:, np.newaxis, :]

        # shape of output: n_samples x n_train
        return (leaves_match / n_by_tree).mean(axis=2)

    def predict(self, X, batch_size=None):
        """
        Predict procedure of GRF.
        It draws a sample based on the frequency of training samples ending in the same leaf as the new sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New samples for which to predict the target values.
        batch_size : int, optional
            Size of the batch to process at once. If None, the entire dataset is processed at once.

        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for each sample in X.
            If the model is a classifier, the output will be class labels.
            If the model is a regressor, the output will be real numbers.
        """
        if batch_size is None:
            weights = self.get_weights(X)
        else:
            list_weights = []
            for batch in np.array_split(X, len(X) // batch_size):
                list_weights.extend(self.get_weights(batch))
            weights = np.array(list_weights)  # n_samples x n_train

        return self.train_y[iterative_random_choice(weights)]


class DrfSk(DrfFitPredictMixin, RandomForestClassifier):
    """Distributional RandomForestClassifier"""


class DrfSkExtraClassifier(DrfFitPredictMixin, ExtraTreesClassifier):
    """Distributional ExtraTreesClassifier"""
