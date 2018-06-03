import numpy as np
from .metrics import r2_score


class LinearRegression:

    def __init__(self):
        """ 初始化Linear Regression模型 """
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """ 正规化方程求解 """
        """ 根据X_train,y_train训练Linear Regression模型 """
        assert X_train.shape[0] == y_train.shape[0], \
            " the size of X_train must be equal to the size of y_train "
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4, epsilon=1e-8):
        """ 梯度下降训练模型 """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to zhe size of y_train"

        # 损失函数 根据theta获取函数值 J(theta) 实际上是 均方误差 MSE
        def J(theta, X_b, y):
            try:
                return ((X_b.dot(theta) - y).dot(X_b.dot(theta) - y)) / len(X_b)
            except:
                return float("inf")

        # 梯度 返回值是一个向量
        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2 / (len(X_b))

        # 梯度下降法
        def gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon):
            i_iter = 0
            theta = initial_theta
            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y))) < epsilon:
                    break
                i_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters, epsilon)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        """ 随机梯度下降训练模型 """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to zhe size of y_train"
        assert n_iters >= 1, "n_iters must greater than 1"

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2

        def sgd(X_b, y, initial_theta, n_iters, t0, t1):
            def learning_rate(t):
                return t0/(t+t1)
            theta = initial_theta
            m = len(X_b)
            for i_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(i_iter * m + i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)
        self.intercept_ = self._theta[0]
        self.coef_= self._theta[1:]
        return self

    def predict(self, X_predict):
        """ 给定测试数据集X_predict,返回预测结果y_predict的结果向量 """
        assert self.intercept_ is not None and self.coef_ is not None, \
            " must fit before predict "
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
