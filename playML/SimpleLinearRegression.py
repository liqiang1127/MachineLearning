import numpy as np
from .metrics import r2_score


class SimpleLinearRegression:

    def __init__(self):
        """ 初始化SimpleLinearRegression1"""
        # 定义实例变量
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """ 根据x_train, y_train训练模型 """
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = (x_train-x_mean).dot(y_train-y_mean)
        d = (x_train-x_mean).dot(x_train-x_mean)
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        """ 给定待预测数据集x_predict, 返回表示x_predict的结果向量 """
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict"
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression()"

