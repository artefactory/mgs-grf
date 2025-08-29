"""

Knn with random ties selection based on scikit-learn implementation.

"""

from collections import namedtuple

import numpy as np
from sklearn.metrics._pairwise_distances_reduction import ArgKminClassMode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import NeighborsBase, _get_weights
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.arrayfuncs import _all_with_any_reduction_axis_1
from sklearn.utils.extmath import weighted_mode
from sklearn.utils.validation import _num_samples, check_is_fitted

ModeResult = namedtuple("ModeResult", ("mode", "count"))


def mode_rand(a, axis):
    """
    Compute the mode of an array along a specified axis, randomly choosing
    among the modes in case of ties.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int
        Axis along which to compute the mode. If None, the mode is computed over the flattened array.

    Returns
    -------
    ModeResult
        A named tuple containing:
            - mode: The mode of the array along the specified axis.
            - count: The count of the mode.

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
    """
    KNN with ties.
    This class extends the KNeighborsClassifier from scikit-learn.
    The main and only modification is to select randomly a sample when ties are encountered.
    """

    _parameter_constraints: dict = {**NeighborsBase._parameter_constraints}
    _parameter_constraints.pop("radius")
    _parameter_constraints.update(
        {"weights": [StrOptions({"uniform", "distance"}), callable, None]}
    )


    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """
        Find the K-neighbors of a point.

        Parameters
        ----------  
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Query points.
        n_neighbors : int, default=None
            Number of neighbors to return for each point. If None, defaults to `n_neighbors` set during initialization.
        return_distance : bool, default=True
            Whether to return distances or not. If True, distances are returned along with indices.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors), optional
            Array of distances to the nearest neighbors for each query point.
            Only returned if `return_distance` is True.
        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Array of indices of the nearest neighbors in the training set.
            If `return_distance` is False, only indices are returned.
        
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
                mode, _ = mode_rand(_y[neigh_ind, k], axis=1)  ### Here modification ##############
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred
