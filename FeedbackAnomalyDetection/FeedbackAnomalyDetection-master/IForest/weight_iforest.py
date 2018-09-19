import numpy as np
import os
import pandas as pd


# from sklearn.ensemble import IsolationForest

learnRate = 0.5


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


def _split_data(X, node):
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

    return left_X, right_X, split_value, feature_id


class Node():
    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None
        self.weight = 1
        self.depth = 0
        self.split_value = 0
        self.indexes = None
        self.isLeafNode = False
        self.feature_id = 0  # 新加的分割特征

    def update(self, X, maxIndex, direction):
        self.weight -= learnRate * direction
        #print(self.weight)
        if self.isLeafNode == False:
            if (self.split_value > X[maxIndex][self.feature_id]):
                self.left.update(X, maxIndex, direction)
            else:
                self.right.update(X, maxIndex, direction)
        elif self.isLeafNode == True:
            return



def iTree(X, max_depth=3):
    ''' construct an isolation tree and returns the number of step required
    to isolate an element.
    A column of index is added to the input matrix X if add_index=True.
    This column is required in the algorithm.
    '''

    n_split = {}

    def getNode(X, count=0):
        n_samples, n_columns = X.shape
        node = Node(data=X)
        node.depth = count


        if count > max_depth:
            for index in X[:, -1]:
                n_split[index] = count
            node.indexes = X[:, -1]
            node.isLeafNode = True
            return node

        if n_samples == 1:
            index = X[0, n_columns - 1]
            n_split[index] = count
            node.indexes = X[0, n_columns - 1]
            node.isLeafNode = True
            return node

        else:

            left_x, right_x, split_value, feature_id = _split_data(X, node)

            node.feature_id = feature_id
            node.split_value = split_value

            n_samples_lX, _ = left_x.shape
            n_samples_rX, _ = right_x.shape

            if n_samples_lX > 0:
                node.left = getNode(left_x,  count + 1)
            if n_samples_rX > 0:
                node.right = getNode(right_x, count + 1)
            return node

    node = getNode(X, 0)

    #printTree(node)
    return node


# use depth as anomaly score
def getlist(node,n_samples):
    n_split = {k: 0 for k in range(n_samples)}
    def iterate(node):
        if node == None:
            return
        if node.isLeafNode == True:
            n_split[node.indexes] = node.depth
        iterate(node.left)
        iterate(node.right)

    iterate(node)

    return n_split

#use weighted path as anomaly score
def getweight(node,n_samples):
    n_weight = {k: 0 for k in range(n_samples)}

    def iterate(node,weight):

        if node == None:
            return
        if node.isLeafNode == True:
           n_weight[node.indexes] = weight
        else:
            iterate(node.left,node.left.weight+weight)
            iterate(node.right,node.right.weight+weight)

    iterate(node,weight=1)
    #print(n_weight)
    return n_weight


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
        self.trees = []
        return

    def fit(self, X):
        n_samples, n_features = X.shape
        if self.sample_size == None:
            self.sample_size = int(n_samples / 2)


        self.trees = [iTree(X[np.random.choice(n_samples,
                                               self.sample_size,
                                               replace=False)],
                            max_depth=self.max_depth)
                      for i in range(self.n_estimators)]

    def get_anomaly_score(self,n_samples):
        treelist = [getweight(self.trees[i],n_samples)
                    for i in range(self.n_estimators)]
        self.path_length_ = {k: None for k in range(n_samples)}  # 这个是字典 总共有n_samples 初始化全部对应 None
        for k in self.path_length_.keys():
            self.path_length_[k] = np.array([tree[k]
                                             for tree in treelist
                                             if k in tree])
        self.path_length_ = np.array([self.path_length_[k].mean() for k in
                                      self.path_length_.keys()])
        #print(self.path_length_)
        self.anomaly_score_ = _anomaly_score(self.path_length_, self.sample_size)
        return self

    def updateweight(self, X, minIndex, direction):
        for i in range(len(self.trees)):
            self.trees[i].update(X, minIndex, direction)
        return


def getfeedback(label,maxIndex):
    if label[maxIndex]==0:
        return 1;
    else:
        return -1



def read_csv(file_name):
    f = open(file_name, 'r')
    content = f.read()
    final_list = list()
    rows = content.split('\n')
    for row in rows:
        final_list.append(row.split(','))
    return final_list

''''
X = np.random.rand(20)  # 全部生成0和1之间
X = X.reshape(5, -1)  # 对数组重新进行整形 10行无所谓几列
n_samples = X.shape[0]

X = np.c_[X, range(n_samples)]  # 增加一列在这里

'''''

#df = pd.read_csv('real_17_feature.csv')
list1= read_csv('real_17_feature.csv')
del list1[0]
list3=[]
label=[]
for x in range(len(list1)):
    list2=[]
    for y in range(len(list1[x])):
        if(y!=1 and y!=0):
            if list1[x][y]!='':
                list2.append(float(list1[x][y]))
            else:
                list2.append(0)
        elif y==1:
            label.append(float(list1[x][y]))
    list3.append(list2)
del list3[len(list3)-1]
X=np.array(list3)
n_samples = X.shape[0]
X = np.c_[X, range(n_samples)]
print(X)
IF = iForest()
IF.fit(X)
IF.get_anomaly_score(n_samples)
trees = IF.trees

instanceiffeedback = np.zeros(len(IF.anomaly_score_),dtype=bool)
direction = 1
####
num1=int(0)
IF.get_anomaly_score(n_samples)
for i in range(500):
    max_score = max(IF.anomaly_score_)
    maxIndex = np.argmax(IF.anomaly_score_)
    feedback = getfeedback(label, maxIndex)
    if feedback==-1:
        num1+=1
    IF.anomaly_score_[maxIndex]=-100

print(num1)


count1=int(0)
count2=int(0)
IF.get_anomaly_score(n_samples)
print(IF.anomaly_score_)
print("mwj")
temp1=[]
temp2=[]
for j in range(50):
    #获取最大的anomaly score 和 最大的 index
    max_score = max(IF.anomaly_score_)
    maxIndex = np.argmax(IF.anomaly_score_)
    feedback = getfeedback(label, maxIndex)
    print(max_score)
    print(maxIndex)
    instanceiffeedback[maxIndex] = True
    if feedback == -1:
        temp1.append( maxIndex)
        direction = 1
        count1 += 1
    else:
        temp2.append(maxIndex)
        direction = -1
        count2+=1
    IF.updateweight(X, maxIndex, direction)
    IF.get_anomaly_score(n_samples)
    for i in range(len(IF.anomaly_score_)):
        if instanceiffeedback[i]==True:
            IF.anomaly_score_[i]=-100
print(temp1)
print(temp2)
print(count1)
print(count2)



num2=int(0)
IF.get_anomaly_score(n_samples)
for i in range(500):
    max_score = max(IF.anomaly_score_)
    maxIndex = np.argmax(IF.anomaly_score_)
    #print(maxIndex)
    feedback = getfeedback(label, maxIndex)
    if feedback==-1:
        num2+=1
    IF.anomaly_score_[maxIndex] = -100

print(num2)






