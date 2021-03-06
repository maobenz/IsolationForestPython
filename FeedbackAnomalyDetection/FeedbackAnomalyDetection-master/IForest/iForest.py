import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import pandas as pd


def _h(i):
    return np.log(i) + 0.5772156649

def _c(n):
    if n > 2:
        h = _h(n - 1)
        return 2 * h - 2 * (n - 1) / n
    if n == 2:
        return 1
    else:
        return 0

def _anomaly_score(score, n_samples):
    score = -score / _c(n_samples)

    return 2 ** score


def _split_data(X):
    ''' split the data in the left and right nodes '''
    n_samples, n_columns = X.shape
    n_features = n_columns - 1
    m = M = 0
    while m == M:
        feature_id = np.random.randint(low=0, high=n_features)
        feature = X[:, feature_id]
        m = feature.min()
        M = feature.max()

    split_value = np.random.uniform(m, M, 1)
    left_X = X[feature <= split_value]
    right_X = X[feature > split_value]
    return left_X, right_X, feature_id, split_value


def iTree(X, max_depth=np.inf):
    ''' construct an isolation tree and returns the number of step required
    to isolate an element.
    A column of index is added to the input matrix X if add_index=True.
    This column is required in the algorithm.
    '''

    n_split = {}

    def iterate(X, count=0):

        n_samples, n_columns = X.shape
        n_features = n_columns - 1

        if count > max_depth:
            for index in X[:, -1]:
                n_split[index] = count
            return

        if n_samples == 1:
            index = X[0, n_columns - 1]
            n_split[index] = count
            return
        else:
            lX, rX, feature_id, split_value = _split_data(X)

            n_samples_lX, _ = lX.shape
            n_samples_rX, _ = rX.shape
            if n_samples_lX > 0:
                iterate(lX, count + 1)
            if n_samples_rX > 0:
                iterate(rX, count + 1)


    iterate(X)
    return n_split


class iForest():
    ''' Class to construct the isolation forest.
    -n_estimators: is the number of trees in the forest,
    -sample_size: is the bootstrap parameter used during the construction
    of the forest,
    -add_index: adds a column of index to the matrix X. This is required and
    add_index can be set to False only if the last column of X contains
    already indeces.
    -max_depth: is the maximum depth of each tree
    '''

    def __init__(self, n_estimators=20, sample_size=256,
                 max_depth=10):
        self.n_estimators = n_estimators
        self.sample_size = sample_size

        self.max_depth = max_depth
        return

    def fit(self, X):
        n_samples, n_features = X.shape
        if self.sample_size == None:
            self.sample_size = int(n_samples / 2)



        trees = [iTree(X[np.random.choice(n_samples,
                                          self.sample_size,
                                          replace=False)],
                       max_depth=self.max_depth)
                 for i in range(self.n_estimators)]

        self.path_length_ = {k: None for k in range(n_samples)}
        for k in self.path_length_.keys():
            self.path_length_[k] = np.array([tree[k]
                                             for tree in trees
                                             if k in tree])
        self.path_length_ = np.array([self.path_length_[k].mean() for k in
                                      self.path_length_.keys()])
        self.anomaly_score_ = _anomaly_score(self.path_length_, self.sample_size)

        return self

X = np.random.rand(10000)
X = X.reshape(1000, -1)
X = np.c_[X, range(1000)]

IF = iForest()
IF.fit(X)
print(IF.anomaly_score_)
