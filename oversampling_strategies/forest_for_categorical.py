import numpy as np

from sklearn.ensemble._forest import ForestClassifier,ForestRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor


class DrfSk(RandomForestClassifier):
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
    
