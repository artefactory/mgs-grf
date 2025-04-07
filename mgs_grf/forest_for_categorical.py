import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)


def iterative_random_choice(probas):
    """
    Function for applying a np.random.choice several times with succesive values of probas
    """
    thresholds = np.random.uniform(size=len(probas))
    cumulative_weights = np.cumsum(probas, axis=1)
    return np.argmax((cumulative_weights.T > thresholds), axis=0)


class DrfFitPredictMixin:
    def fit(self, X, y, sample_weight=None):
        super().fit(X=X, y=y, sample_weight=sample_weight)
        self.train_y = y
        self.train_samples_leaves = (
            super().apply(X).astype(np.int32)
        )  # train_samples_leaves: size n_train x n_trees

    def get_weights(self, X):
        leafs_by_sample = (
            super().apply(X).astype(np.int32)
        )  # taille n_samples x n_trees
        leaves_match = np.array(
            [leafs_by_sample[i] == self.train_samples_leaves for i in range(len(X))]
        )
        #### NEEEW ####
        # start_old = time.time()
        ### taille n_samples x n_train x n_trees
        # leafs_by_sample_repeat = np.tile(leafs_by_sample.T, (len(self.train_samples_leaves), 1, 1)).T
        # print("repeat : ", time.time()-start_old)
        # leaves_match = leafs_by_sample_repeat == self.train_samples_leaves.T
        ### taille n_samples x n_trees x n_train
        # print("Weights leaves match NEW : ", time.time()-start_old)
        # n_by_tree = leaves_match.sum(axis=2)
        # w = (leaves_match / n_by_tree[:,:, np.newaxis]).mean(axis=1) # taille n_samples x n_train
        #### END NEEW #####

        n_by_tree = leaves_match.sum(axis=1)[:, np.newaxis, :]
        # leaves_match = leaves_match.astype(np.float16)
        # leaves_match /= n_by_tree
        # w = leaves_match.mean(axis=2) # taille n_samples x n_train
        # print('b')
        w = (leaves_match / n_by_tree).mean(axis=2)  # taille n_samples x n_train
        return w

    def predict(self, X, batch_size=None):
        if batch_size is None:
            weights = self.get_weights(X)
        else:
            list_weights = []
            for batch in np.array_split(X, len(X) // batch_size):
                list_weights.extend(self.get_weights(batch))
            weights = np.array(list_weights)  # n_samples x n_train
        res = self.train_y[iterative_random_choice(weights)]
        return res


class DrfSk(DrfFitPredictMixin, RandomForestClassifier):
    """Blabla"""


class DrfSkRegressor(DrfFitPredictMixin, RandomForestRegressor):
    """Blabla"""


class DrfSkExtraClassifier(DrfFitPredictMixin, ExtraTreesClassifier):
    """Blabla"""


class DrfSkExtraRegressor(DrfFitPredictMixin, ExtraTreesRegressor):
    """Blabla"""


