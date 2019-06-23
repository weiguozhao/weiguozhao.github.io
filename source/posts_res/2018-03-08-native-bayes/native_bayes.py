# coding:utf-8

import numpy as np
import time
from datetime import timedelta
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def time_consume(s_t):
    diff = time.time()-s_t
    return timedelta(seconds=int(diff))


class NativeBayes(object):
    def __init__(self, class_num=-1, features_dim=-1):
        self.class_num = class_num
        self.features_dim = features_dim

    def fit(self, X, y):
        self.nlen = X.shape[0]
        # shape meaning: class_num
        prior_prob = np.zeros(shape=self.class_num)
        # shape meaning: class_num, feature_dim, feature_value
        conditional_prob = np.zeros(shape=(self.class_num, self.features_dim, 2))

        for i in range(self.nlen):
            prior_prob[y[i]] += 1.0
            for j in range(self.features_dim):
                conditional_prob[y[i]][j][X[i][j]] += 1.0

        for i in range(self.class_num):
            for j in range(self.features_dim):
                p_0 = conditional_prob[i][j][0]
                p_1 = conditional_prob[i][j][1]
                prob_0 = p_0 / (p_0 + p_1)
                prob_1 = p_1 / (p_0 + p_1)
                conditional_prob[i][j][0] = prob_0
                conditional_prob[i][j][1] = prob_1

        self.prior_prob = prior_prob
        self.conditional_prob = conditional_prob

    def __calculate_prob__(self, sample, label):
        prob = int(self.prior_prob[label])
        for i in range(self.features_dim):
            prob *= self.conditional_prob[label][i][sample[i]]
        return prob

    def predict(self, X):
        if self.class_num == -1:
            raise ValueError("Please fit first.")
        y_pred = np.zeros(shape=X.shape[0])
        for i in range(X.shape[0]):
            label = 0
            prob = self.__calculate_prob__(X[i], 0)
            for j in range(1, self.class_num):
                this_prob = self.__calculate_prob__(X[i], j)
                if prob < this_prob:
                    prob = this_prob
                    label = j
            y_pred[i] = label
        return y_pred

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


def load_data():
    """normalize value of data to {0, 1}"""
    data, target = load_breast_cancer(return_X_y=True)
    nlen = data.shape[0]
    ndim = data.shape[1]
    X = np.zeros(shape=data.shape, dtype=np.int32)
    for i in range(ndim):
        mean = np.mean(data[:, i])
        for j in range(nlen):
            if data[j][i] > mean:
                X[j][i] = 1
    return X, target


if __name__ == '__main__':
    start_time = time.time()

    data, target = load_data()
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=2048, shuffle=True)

    ml = NativeBayes(class_num=2, features_dim=30)
    ml.fit(train_x, train_y)
    y_pred = ml.predict(test_x)
    accuracy = ml.accuracy(test_y, y_pred)
    print("Accuracy is ", accuracy)
    print("Timeusage: %s" % (time_consume(start_time)))
