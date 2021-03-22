# coding:utf-8
import numpy as np
import pandas as pd

class DecisionTree(object):
    def __init__(self, feature_names, threshold, principle="information gain"):
        self.feature_names = feature_names
        self.threshold = threshold
        self.principle = principle
    # formula 5.7
    def __calculate_entropy__(self, y):
        datalen = len(y)
        labelprob = {l: 0 for l in set(y)}
        entropy = 0.0
        for l in y:
            labelprob[l] += 1
        for l in labelprob.keys():
            thisfrac = labelprob[l] / datalen
            entropy -= thisfrac * np.log2(thisfrac)
        return entropy
    # formula 5.8
    def __calculate_conditional_entropy__(self, X, y, axis):
        datalen = len(y)
        featureset = set([x[axis] for x in X])
        sub_y = {f:list() for f in featureset}
        for i in range(datalen):
            sub_y[X[i][axis]].append(y[i])
        conditional_entropy = 0.0
        for key in sub_y.keys():
            prob = len(sub_y[key]) / datalen
            entropy = self.__calculate_entropy__(sub_y[key])
            conditional_entropy += prob * entropy
        return conditional_entropy
    # formula 5.9
    def calculate_information_gain(self, X, y, axis):
        hd = self.__calculate_entropy__(y)
        hda = self.__calculate_conditional_entropy__(X, y, axis)
        gda = hd - hda
        return gda

    def __most_class__(self, y):
        labelset = set(y)
        labelcnt = {l:0 for l in labelset}
        for y_i in y:
            labelcnt[y_i] += 1
        st = sorted(labelcnt.items(), key=lambda x: x[1], reverse=True)
        return st[0][0]
    # formula 5.10
    def calculate_information_gain_ratio(self, X, y, axis):
        gda = self.calculate_information_gain(X, y, axis)
        had = self.__calculate_entropy__(X[:, axis])
        grda = gda / had
        return grda

    def __split_dataset__(self, X, y, axis, value):
        rstX = list()
        rsty = list()
        for i in range(len(X)):
            if X[i][axis] == value:
                tmpfeature = list(X[i][:axis])
                tmpfeature.extend(list(X[i][axis+1:]))
                rstX.append(tmpfeature)
                rsty.append(y[i])
        return np.asarray(rstX), np.asarray(rsty)

    def __best_split_feature__(self, X, y, feature_names):
        best_feature = -1
        max_principle = -1.0
        for feature_n in feature_names:
            axis = feature_names.index(feature_n)
            if self.principle == "information gain":
                this_principle = self.calculate_information_gain(X, y, axis)
            else:
                this_principle = self.calculate_information_gain_ratio(X, y, axis)
            print("%s\t%f\t%s" % (feature_n, this_principle, self.principle))
            if this_principle > max_principle:
                best_feature = axis
                max_principle = this_principle
        print("-----")
        return best_feature, max_principle

    def _fit(self, X, y, feature_names):
        # 所有实例属于同一类
        labelset = set(y)
        if len(labelset) == 1:
            return labelset.pop()
        # 如果特征集为空集，置T为单结点树，实例最多的类作为该结点的类，并返回T
        if len(feature_names) == 0:
            return self.__most_class__(y)
        # 计算准则,选择特征
        best_feature, max_principle = self.__best_split_feature__(X, y, feature_names)
        # 如果小于阈值，置T为单结点树，实例最多的类作为该结点的类，并返回T
        if max_principle < self.threshold:
            return self.__most_class__(y)

        best_feature_label = feature_names[best_feature]
        del feature_names[best_feature]
        tree = {best_feature_label: {}}

        bestfeature_values = set([x[best_feature] for x in X])
        for value in bestfeature_values:
            sub_X, sub_y = self.__split_dataset__(X, y, best_feature, value)
            tree[best_feature_label][value] = self._fit(sub_X, sub_y, feature_names)
        return tree

    def fit(self, X, y):
        feature_names = self.feature_names[:]
        self.tree = self._fit(X, y, feature_names)

    def _predict(self, tree, feature_names, x):
        firstStr = list(tree.keys())[0]
        secondDict = tree[firstStr]
        featIndex = feature_names.index(firstStr)
        key = x[featIndex]
        valueOfFeat = secondDict[key]
        if isinstance(valueOfFeat, dict):
            classLabel = self._predict(valueOfFeat, feature_names, x)
        else:
            classLabel = valueOfFeat
        return classLabel

    def predict(self, X):
        preds = list()
        for x in X:
            preds.append(self._predict(self.tree, self.feature_names, x))
        return preds

    def output_tree(self):
        import treePlot     # cite: https://gitee.com/orayang_admin/ID3_decisiontree/tree/master
        import importlib
        importlib.reload(treePlot)
        treePlot.createPlot(self.tree)

def load_data():
    dt = pd.read_csv("./credit.csv")    # from lihang - "statistic learning method" - page59, table 5.1
    # dt = pd.read_csv("./titanic.csv")
    data = dt.values
    feature_names = dt.columns[:-1] # delete label column
    return data, list(feature_names)

def run_ID3():
    data, feature_names = load_data()
    print("ID3 Descision Tree ... ")
    ml = DecisionTree(feature_names=feature_names, threshold=0, principle="information gain")
    ml.fit(data[:, :-1], data[:, -1])
    test = [["mid", "yes", "no", "good"]]
    preds = ml.predict(test)
    print("ID3 predict:", preds)
    ml.output_tree()

def run_C45():
    data, feature_names = load_data()
    print("C45 Descision Tree ... ")
    ml = DecisionTree(feature_names=feature_names, threshold=0, principle="information gain ratio")
    ml.fit(data[:, :-1], data[:, -1])
    test = [["mid", "yes", "no", "good"]]
    preds = ml.predict(test)
    print("C45 predict:", preds)
    ml.output_tree()

if __name__ == '__main__':
    run_ID3()
    # run_C45()
