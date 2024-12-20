import numpy as np

from sklearn.ensemble._forest import ForestClassifier,ForestRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor

from collections import namedtuple ## KNN
from sklearn.utils.arrayfuncs import _all_with_any_reduction_axis_1
from sklearn.utils.extmath import weighted_mode
from sklearn.utils.validation import _num_samples, check_is_fitted
from sklearn.metrics._pairwise_distances_reduction import ArgKminClassMode
from sklearn.neighbors._base import _get_weights,NeighborsBase
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils._param_validation import StrOptions


class DrfSk(RandomForestClassifier):
    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **DecisionTreeClassifier._parameter_constraints,
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super(ForestClassifier,self).__init__(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.monotonic_cst = monotonic_cst
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        self.trained_X = X
        self.trained_y = y
        super().fit(X=X,y=y,sample_weight=sample_weight)
        self.train_samples_leaves = []
        for tree in self.estimators_:
            self.train_samples_leaves.append(tree.apply(self.trained_X)) 

    def get_weights(self,x):   
        n_tree = len(self.estimators_)
        w=np.zeros((len(self.trained_X),))
        for t,tree in enumerate(self.estimators_):
            #train_samples_leaves = tree.apply(self.trained_X)
            train_samples_leaves = self.train_samples_leaves[t]
            x_leaf = tree.apply(x)[0]
            indices_train_samples_in_same_leaf = np.where(train_samples_leaves==x_leaf)[0]
            n_leaves_in = len(indices_train_samples_in_same_leaf)
            if n_leaves_in != 0:
                for idx in indices_train_samples_in_same_leaf:
                    w[idx] = w[idx] + 1/(n_tree*n_leaves_in)
        return w

    def predict(self,X):
        #weights_all = np.apply_along_axis(self.get_weights,0,X,{'self':self})
        size_train = len(self.trained_X)
        list_index_train_X = np.arange(start=0,stop=size_train,step=1)
        y_pred = np.zeros((len(X),self.trained_y.shape[1]),dtype=self.trained_y.dtype)
        for i in range(len(X)):
            x = X[i,:].reshape(1, -1)
            w = self.get_weights(x)
            selected_index = np.random.choice(a=list_index_train_X,size=1,replace=False,p=w)
            y_pred[i] = self.trained_y[selected_index]
            
        return np.array(y_pred)
    



class DrfSkRegressor(RandomForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super(ForestRegressor,self).__init__(
            estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

    def fit(self, X, y, sample_weight=None):
        self.trained_X = X
        self.trained_y = y
        super().fit(X=X,y=y,sample_weight=sample_weight)
        self.train_samples_leaves = []
        for tree in self.estimators_:
            self.train_samples_leaves.append(tree.apply(self.trained_X)) 

    def get_weights(self,x):   
        n_tree = len(self.estimators_)
        w=np.zeros((len(self.trained_X),))
        for t,tree in enumerate(self.estimators_):
            #train_samples_leaves = tree.apply(self.trained_X)
            train_samples_leaves = self.train_samples_leaves[t]
            x_leaf = tree.apply(x)[0]
            indices_train_samples_in_same_leaf = np.where(train_samples_leaves==x_leaf)[0]
            n_leaves_in = len(indices_train_samples_in_same_leaf)
            if n_leaves_in != 0:
                for idx in indices_train_samples_in_same_leaf:
                    w[idx] = w[idx] + 1/(n_tree*n_leaves_in)
        return w

    def predict(self,X):
        #weights_all = np.apply_along_axis(self.get_weights,0,X,{'self':self})
        size_train = len(self.trained_X)
        list_index_train_X = np.arange(start=0,stop=size_train,step=1)
        y_pred = np.zeros((len(X),self.trained_y.shape[1]))
        for i in range(len(X)):
            x = X[i,:].reshape(1, -1)
            w = self.get_weights(x)
            selected_index = np.random.choice(a=list_index_train_X,size=1,replace=False,p=w)
            y_pred[i] = self.trained_y[selected_index]

        return np.array(y_pred)
    

ModeResult = namedtuple('ModeResult', ('mode', 'count'))
def mode_rand(a, axis):
    in_dims = list(range(a.ndim))
    a_view = np.transpose(a, in_dims[:axis] + in_dims[axis+1:] + [axis])

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
    _parameter_constraints: dict = {**NeighborsBase._parameter_constraints}
    _parameter_constraints.pop("radius")
    _parameter_constraints.update(
        {"weights": [StrOptions({"uniform", "distance"}), callable, None]}
    )

    def __init__(self,
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
        super().__init__(n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        if n_neighbors == None:
            n_neighbors = self.n_neighbors +1
        else:
            n_neighbors = n_neighbors+1
            
        if return_distance:
            neigh_dist, neigh_ind = super().kneighbors(X=X, n_neighbors=n_neighbors,return_distance=return_distance)
            return neigh_dist[:,1:], neigh_ind[:,1:]
        else:
            neigh_ind = super().kneighbors(X=X, n_neighbors=n_neighbors,return_distance=return_distance)
            return  neigh_ind[:,1:]
        
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
                mode, _ = mode_rand(_y[neigh_ind, k], axis=1) ## Here modification 
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred
    
