import numpy as np
import os
#from sklearn.ensemble import IsolationForest

learnRate=0.1  #the rate of learning
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


def _split_data(X,node):
    ''' split the data in the left and right nodes '''

    n_samples, n_columns = X.shape
    n_features = n_columns - 1
    m = M = 0
    while m == M:
        feature_id = np.random.randint(low=0, high=n_features)
        print(feature_id)
        feature = X[:, feature_id]
        m = feature.min()
        M = feature.max()

    split_value = np.random.uniform(m, M, 1)
    left_X = X[feature <= split_value]
    right_X = X[feature > split_value]

    return left_X, right_X, split_value,feature_id


class Node():
    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None
        self.weight = 1
        self.allWeight = 1
        self.depth = 0
        self.split_value = 0
        self.indexes = None
        self.isLeafNode = False
        self.feature_id=0;   #新加的分割特征

    def update(self,X,minIndex,direction):
        self.weight += learnRate * direction
        if self.isLeafNode==False:
            if(self.split_value>X[minIndex][self.feature_id]):
                self.left.update(X,minIndex,direction)
            else:
                self.right.update(X,minIndex,direction)
        elif self.isLeafNode==True:
            return ;




def iTree(X, max_depth=3):
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
            for index in X[:, -1]:
                n_split[index] = count
            node.indexes = X[:,-1]
            node.isLeafNode = True
            #print('step1', node.indexes)
            return node

        if n_samples == 1:
            index = X[0, n_columns - 1]
            n_split[index] = count
            node.indexes = X[0, n_columns - 1]
            #print('step2', node.indexes)
            node.isLeafNode = True
            return node

        else:

            left_x, right_x, split_value,feature_id = _split_data(X,node)

            node.feature_id=feature_id
            node.split_value = split_value

            n_samples_lX, _ = left_x.shape
            n_samples_rX, _ = right_x.shape

            if n_samples_lX > 0:
                node.left = getNode(left_x, node.allWeight, count + 1)
            if n_samples_rX > 0:
                node.right = getNode(right_x, node.allWeight, count + 1)
            return node

    node = getNode(X, 0)
    #print(node.isLeafNode)
    #printTree(node)
    return node



def getlist(node):
    n_split = {}
    def iterate(node):
        global count1
        if node==None:
            return
        if node.isLeafNode ==True:
            n_split[node.indexes]=node.depth
        iterate(node.left)
        iterate(node.right)
    iterate(node)
    return n_split


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

    def __init__(self, n_estimators=20, sample_size=256, add_index=True,
                 max_depth=100):
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.add_index = add_index
        self.max_depth = max_depth
        self.trees=[]
        return

    def fit(self, X):
        n_samples, n_features = X.shape
        if self.sample_size == None:
            self.sample_size = int(n_samples / 2)

        if self.add_index:
            X = np.c_[X, range(n_samples)]
        self.trees= [iTree(X[np.random.choice(n_samples,
                                          self.sample_size,
                                          replace=False)],
                     max_depth=self.max_depth)
                for i in range(self.n_estimators)]
        treelist =[getlist(self.trees[i])
                     for i in range(self.n_estimators)]
        self.path_length_ = {k: None for k in range(n_samples)}  #这个是字典 总共有n_samples 初始化全部对应 None
        for k in self.path_length_.keys():
            self.path_length_[k] = np.array([tree[k]
                                             for tree in treelist
                                             if k in tree])
        self.path_length_ = np.array([self.path_length_[k].mean() for k in
                                      self.path_length_.keys()])
        print(self.path_length_)
        self.anomaly_score_ = _anomaly_score(self.path_length_, self.sample_size)
        return self

    def updateweight(self,X,minIndex,direction):
        for i in range(len(self.trees)):
            self.trees[i].update(X,minIndex,direction)
        return

    def updateweightlog(X, minIndex,direction):
        return

    def updateweightlogister(X, minIndex,direction):
        return



def getfeedback(X):
    return -1

if __name__ == '__main__':

    X = np.random.rand(10000)   #全部生成0和1之间
    X = X.reshape(1000, -1)    #对数组重新进行整形 10行无所谓几列
    X = np.c_[X, range(1000)]   #增加一列在这里
    node=iTree(X)
    IF = iForest()
    IF.fit(X)
    print(IF.anomaly_score_)
    instanceiffeedback=[]   #这个数据是否已经被标记
    direction=1
    typeloss =int(input())
    os.system("pause")
    print(typeloss)
    for i in range(len(IF.anomaly_score_)):
        instanceiffeedback.append(False)  #初始化
    for j in range(100):
        min=100
        minIndex=-1
        for i in range(len(IF.anomaly_score_)):
            if min>IF.anomaly_score_[i] and instanceiffeedback[i]==False:
                min=IF.anomaly_score_[i]
                minIndex=i
        #print(len(IF.anomaly_score_))
        feedback=getfeedback(X[minIndex])
        instanceiffeedback[minIndex]=True
        if feedback==-1:
            direction = -1
        else:
            direction=1;
        if(typeloss==1):
            IF.updateweight(X, minIndex,direction)
        elif typeloss==2:
            IF.updateweightlog(X, minIndex,direction)
        elif typeloss==3:
            IF.updateweightlogister(X, minIndex,direction)



