import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:
    def __init__(self, k):
        """ 初始化kNN分类器 """
        assert k >= 1, " k must be valid "
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """ 根据参数训练模型 """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_trian"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be as least k"
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self,X_predict):
        """ 预测 """
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be equal to X_train"
        y_predict = [self._predict(x_predict) for x_predict in X_predict]
        return np.array(y_predict)

    def _predict(self, x_predict):
        """ 给单个预测数据x_predict,返回x的预测结果 """
        assert x_predict.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"
        # kNN过程
        diatances = [sqrt(np.sum((x_train - x_predict)**2)) for x_train in self._X_train]
        nearest = np.argsort(diatances)
        votes = Counter(self._y_train[i] for i in nearest[:self.k])
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k
