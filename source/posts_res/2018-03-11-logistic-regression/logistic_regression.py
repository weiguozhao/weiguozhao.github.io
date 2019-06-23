# coding:utf-8

import time
import numpy as np
from datetime import timedelta
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def time_consume(s_t):
    diff = time.time()-s_t
    return timedelta(seconds=int(diff))


class BinaryLR(object):
    def __init__(self):
        self.weight = None
        self.learning_rate = 0.001
        self.max_iteration = 3000

    def predict_single_sample(self, feature):
        feature = list(feature)
        feature.append(1.0)
        wx = np.sum(np.matmul(self.weight, feature))
        exp_wx = np.exp(wx)
        if exp_wx/(1+exp_wx) > 0.5:
            return 1
        else:
            return 0

    def fit(self, X, y):
        self.nlen = X.shape[0]
        self.ndim = X.shape[1]
        self.weight = np.zeros(shape=self.ndim+1) # add bias to weight

        correct = 0
        exec_times = 0
        while exec_times < self.max_iteration:
            index = np.random.randint(0, self.nlen)
            feature = list(X[index])
            label = self.predict_single_sample(feature)

            if label == y[index]:
                correct += 1
                if correct > self.max_iteration:
                    break
                continue

            exec_times += 1
            correct = 0

            feature.append(1.0)
            wx = np.sum(np.matmul(self.weight, feature))
            exp_wx = np.exp(wx)

            # update weight
            for i in range(self.weight.shape[0]):
                self.weight[i] -= self.learning_rate * (label - exp_wx / (1.0 + exp_wx)) * feature[i]

            if exec_times % 100 == 0:
                print("Times:%d TrainAcc:%.4f Timeusage:%s" % (exec_times, self.accuracy(train_y, self.predict(train_x)), time_consume(start_time)))


    def predict(self, X):
        if self.weight is None:
            raise ValueError("Please train model first.")

        labels = list()
        for i in range(X.shape[0]):
            d = X[i]
            labels.append(self.predict_single_sample(d))
        return labels

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


def load_dataset():
    data = load_breast_cancer()
    return data.data, data.target


if __name__ == '__main__':
    start_time = time.time()

    data, target = load_dataset()
    print("Data Shape:", data.shape, target.shape)

    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=1024, shuffle=True)

    ml = BinaryLR()
    ml.fit(train_x, train_y)
    y_pred = ml.predict(test_x)
    accuracy = ml.accuracy(test_y, y_pred)
    print("Accuracy is ", accuracy)
    print("Timeusage: %.2f s" % (time.time()-start_time))
