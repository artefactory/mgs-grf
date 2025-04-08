"""Generlized Random Forest for categorical variables."""
from collections import namedtuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics._pairwise_distances_reduction import ArgKminClassMode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import NeighborsBase, _get_weights
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.arrayfuncs import _all_with_any_reduction_axis_1
from sklearn.utils.extmath import weighted_mode
from sklearn.utils.validation import _num_samples, check_is_fitted


def iterative_random_choice(probas):
    """
    Function for applying a np.random.choice several times with succesive values of probas
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
        super().fit(X=X, y=y, sample_weight=sample_weight)
        self.train_y = y
        self.train_samples_leaves = (
            super().apply(X).astype(np.int32)
        )  # train_samples_leaves: size n_train x n_trees

    def get_weights(self, X):
        leafs_by_sample = super().apply(X).astype(np.int32)  # taille n_samples x n_trees
        leaves_match = np.array(
            [leafs_by_sample[i] == self.train_samples_leaves for i in range(len(X))]
        )
        n_by_tree = leaves_match.sum(axis=1)[:, np.newaxis, :]
        # leaves_match = leaves_match.astype(np.float16)
        # leaves_match /= n_by_tree
        # w = leaves_match.mean(axis=2) # taille n_samples x n_train
        return (leaves_match / n_by_tree).mean(axis=2)  # taille n_samples x n_train

    def predict(self, X, batch_size=None):
        """
        Preditc procedure of GRF.
        batch_size : int
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
    """Blabla"""


class DrfSkRegressor(DrfFitPredictMixin, RandomForestRegressor):
    """Blabla"""


class DrfSkExtraClassifier(DrfFitPredictMixin, ExtraTreesClassifier):
    """Blabla"""


class DrfSkExtraRegressor(DrfFitPredictMixin, ExtraTreesRegressor):
    """Blabla"""


ModeResult = namedtuple("ModeResult", ("mode", "count"))


def mode_rand(a, axis):
    """_summary_.

    Parameters
    ----------
    a : _type_
        _description_
    axis : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    in_dims = list(range(a.ndim))
    a_view = np.transpose(a, in_dims[:axis] + in_dims[axis + 1 :] + [axis])

    inds = np.ndindex(a_view.shape[:-1])
    modes = np.empty(a_view.shape[:-1], dtype=a.dtype)
    counts = np.zeros(a_view.shape[:-1], dtype=int)

    for ind in inds:
        vals, cnts = np.unique(a_view[ind], return_counts=True)
        maxes = np.where(cnts == cnts.max())  # Here's the change
        modes[ind], counts[ind] = vals[np.random.choice(maxes[0])], cnts.max()

    newshape = list(a.shape)
    newshape[axis] = 1
    return ModeResult(modes.reshape(newshape), counts.reshape(newshape))


class KNNTies(KNeighborsClassifier):
    """KNN with ties."""
    _parameter_constraints: dict = {**NeighborsBase._parameter_constraints}
    _parameter_constraints.pop("radius")
    _parameter_constraints.update(
        {"weights": [StrOptions({"uniform", "distance"}), callable, None]}
    )

    def ____init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        """_summary_.

        Parameters
        ----------
        n_neighbors : int, optional
            _description_, by default 5
        weights : str, optional
            _description_, by default "uniform"
        algorithm : str, optional
            _description_, by default "auto"
        leaf_size : int, optional
            _description_, by default 30
        p : int, optional
            _description_, by default 2
        metric : str, optional
            _description_, by default "minkowski"
        metric_params : _type_, optional
            _description_, by default None
        n_jobs : _type_, optional
            _description_, by default None
        """
        super().__init__(
            n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """_summary_.

        Parameters
        ----------
        X : _type_, optional
            _description_, by default None
        n_neighbors : _type_, optional
            _description_, by default None
        return_distance : bool, optional
            _description_, by default True

        Returns
        -------
        _type_
            _description_
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors + 1
        else:
            n_neighbors = n_neighbors + 1

        if return_distance:
            neigh_dist, neigh_ind = super().kneighbors(
                X=X, n_neighbors=n_neighbors, return_distance=return_distance
            )
            return neigh_dist[:, 1:], neigh_ind[:, 1:]
        neigh_ind = super().kneighbors(
                X=X, n_neighbors=n_neighbors, return_distance=return_distance
            )
        return neigh_ind[:, 1:]

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        check_is_fitted(self, "_fit_method")
        if self.weights == "uniform":
            if self._fit_method == "brute" and ArgKminClassMode.is_usable_for(
                X, self._fit_X, self.metric
            ):
                probabilities = self.predict_proba(X)
                if self.outputs_2d_:
                    return np.stack(
                        [
                            self.classes_[idx][np.argmax(probas, axis=1)]
                            for idx, probas in enumerate(probabilities)
                        ],
                        axis=1,
                    )
                return self.classes_[np.argmax(probabilities, axis=1)]
            # In that case, we do not need the distances to perform
            # the weighting so we do not compute them.
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_queries = _num_samples(X)
        weights = _get_weights(neigh_dist, self.weights)
        if weights is not None and _all_with_any_reduction_axis_1(weights, value=0):
            raise ValueError(
                "All neighbors of some sample is getting zero weights. "
                "Please modify 'weights' to avoid this case if you are "
                "using a user-defined function."
            )

        y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = mode_rand(_y[neigh_ind, k], axis=1)  ## Here modification
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred
