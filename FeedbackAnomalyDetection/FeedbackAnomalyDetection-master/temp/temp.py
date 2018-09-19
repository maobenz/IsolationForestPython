import numpy as np


def _h(i):
    return np.log(i) + 0.5772156649


def _c(n):
    if n > 2:
        h = _h(n - 1)
        return 2 * h - 2 * (n - 1) / n
    # if n == 2:
    #     return 1
    else:
        return 1


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

    return left_X, right_X, split_value


class Node():
    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None
        self.weight = 1  #节点的权重
        self.allWeight = 1  #继承过来的权重+自身权重
        self.depth = 0
        self.split_value = 0
        self.indexes = None
        self.isLeafNode = False


def iTree(X, max_depth=4):
    ''' construct an isolation tree and returns the number of step required
    to isolate an element.
    A column of index is added to the input matrix X if add_index=True.
    This column is required in the algorithm.
    '''

    n_split = {}

    def getNode(X, weight, count=0):
        n_samples, n_columns = X.shape
        node = Node(data=X)
        node.depth = count
        node.allWeight=node.weight + weight

        if count > max_depth:
            node.indexes = X[:, -1]
            node.isLeafNode = True
            print('step1', node.indexes)
            return node

        if n_samples == 1:
            node.indexes = X[0, n_columns - 1]
            print('step2', node.indexes)
            node.isLeafNode = True
            return node

        else:

            left_x, right_x, split_value = _split_data(X)
            node.split_value = split_value

            n_samples_lX, _ = left_x.shape
            n_samples_rX, _ = right_x.shape

            if n_samples_lX > 0:
                node.left = getNode(left_x, node.allWeight, count + 1)
            if n_samples_rX > 0:
                node.right = getNode(right_x, node.allWeight, count + 1)
            return node

    node = getNode(X, 0)
    print(node.isLeafNode)
    printTree(node)
    return node


def printTree(node):
    if node == None:
        return
    if node.isLeafNode == True:
        print(node.indexes, node.depth)
    else:
        printTree(node.left)
        printTree(node.right)


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

    def __init__(self, n_estimators=1, sample_size=256, add_index=True,
                 max_depth=100):
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.add_index = add_index
        self.max_depth = max_depth
        return

    def fit(self, X):
        n_samples, n_features = X.shape
        if self.sample_size == None:
            self.sample_size = int(n_samples / 2)

        if self.add_index:
            X = np.c_[X, range(n_samples)]

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


X = np.random.rand(100)
X = X.reshape(10, -1)
X = np.c_[X, range(10)]
iTree(X)
# IF = iForest()
# IF.fit(X)
# print(IF.anomaly_score_)
