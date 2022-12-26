from collections import Counter as cr

import numpy as np


class Node:
    def __init__(self, features=None, threshold=None, right=None, left=None, *, value=None):
        self.__value = value
        self.__threshold = threshold
        self.__features = features
        self.__left = left
        self.__right = right

    def set_value(self, vlaue):
        self.__value = vlaue

    def set_threshold(self, threshold):
        self.__threshold = threshold

    def set_feature(self, features):
        self.__features = features

    def set_left(self, left):
        self.__left = left

    def set_right(self, right):
        self.__right = right

    def is_leaf(self):
        return self.__value is not None

    def get_value(self):
        return self.__value

    def get_feature(self):
        return self.__features

    def get_threshold(self):
        return self.__threshold

    def get_left(self):
        return self.__left

    def get_right(self):
        return self.__right

    def __str__(self):
        Str = "value = " + str(self.__value)
        Str += "\nfeatures = " + str(self.__features)
        Str += "\nthreshold = " + str(self.__threshold)
        Str += "\nleft node\n" + str(self.__left)
        Str += "\nright node\n" + str(self.__right)
        return Str


class DecisionTree():
    def __init__(self, min_splits=2, max_depth=100, features_num=None, root=None):
        self.__root = root
        self.__features_num = features_num
        self.__max_depth = max_depth
        self.__min_splits = min_splits

    def fit(self, X, Y):
        self.__features_num = X.shape[1] if not self.__features_num else min(self.__features_num, X.shape[1])
        self.__root = self._build_tree(X, Y)

    def _build_tree(self, X, Y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(Y))

        if depth >= self.__max_depth or n_labels == 1 or n_samples < self.__min_splits:
            leaf_value = self._most_common_label(Y)
            return Node(value=leaf_value)

        features_idx = np.random.choice(n_features, self.__features_num, replace=False)

        best_thresh, best_features = self._best_divide(X, Y, features_idx)

        left_indx, right_indx = self._divide(X[:, best_features], best_thresh)
        left = self._build_tree(X[left_indx, :], Y[left_indx], depth + 1)
        right = self._build_tree(X[right_indx, :], Y[right_indx], depth + 1)
        return Node(best_features, best_thresh, right, left)

    def _most_common_label(self, Y):
        counter = cr(Y)
        value = counter.most_common(1)[0][0]
        return value

    def _best_divide(self, X, Y, feat_idx):
        best_gain = -1
        divide_indx, divide_thresh = None, None

        for i in feat_idx:
            x_col = X[:, i]
            threshold = np.unique(x_col)

            for thre in threshold:
                gain = self._info_gain(x_col, Y, thre)
                if gain > best_gain:
                    best_gain = gain
                    divide_indx = i
                    divide_thresh = thre

        return divide_thresh, divide_indx

    def _info_gain(self, X_col, Y, threshold):
        P_entr = self._entropy(Y)

        left_indx, rigth_indx = self._divide(X_col, threshold)

        if len(left_indx) == 0 or len(rigth_indx) == 0:
            return 0

        n = len(Y)
        n_s_l, n_s_r = len(left_indx), len(rigth_indx)
        e_l, e_r = self._entropy(Y[left_indx]), self._entropy(Y[rigth_indx])
        C_entr = (n_s_l / n) * e_l + (n_s_r / n) * e_r

        information_gain = P_entr - C_entr
        return information_gain

    def _entropy(self, Y):
        count_x = np.bincount(Y)
        prop_x = count_x / len(Y)
        return -np.sum([p * np.log(p) for p in prop_x if p > 0])

    def _divide(self, X_col, threshold):
        left_indx = np.argwhere(X_col <= threshold).flatten()
        right_indx = np.argwhere(X_col > threshold).flatten()
        return left_indx, right_indx

    def predict(self, X):
        results = [self._simulate_tree(x, self.__root) for x in X]
        return np.array(results)

    def _simulate_tree(self, x, node):
        if node.is_leaf():
            return node.get_value()

        if x[node.get_feature()] <= node.get_threshold():
            return self._simulate_tree(x, node.get_left())
        else:
            return self._simulate_tree(x, node.get_right())

    def get_root(self):
        return self.__root

    def get_featuers_num(self):
        return self.__features_num

    def get_max_depth(self):
        return self.__max_depth

    def get_min_splits(self):
        return self.__min_splits

    def __str__(self):
        Str = "root = " + str(self.__root)
        Str += "\ndepth = " + str(self.__max_depth)
        Str += "\nfeatures num = " + str(self.__features_num)
        Str += "\nmin splits = " + str(self.__min_splits)
        return Str
