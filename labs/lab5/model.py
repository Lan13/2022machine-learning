import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from itertools import product
from skrebate import ReliefF
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class LogisticRegression:
    def __init__(self, penalty="l2", lr=0.01, tol=1e-7, max_iter=1e4):
        self.thetas = []
        self.train_loss = []
        self.penalty = penalty
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        err_msg = "vpenalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg

    def sigmoid(self, x):
        return 1 / (1 + (np.exp(-x)))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        y = np.expand_dims(y, axis=1)

        for i in np.unique(y):
            y_ova = np.where(y == i, 1, 0)
            theta = np.zeros((X.shape[1], 1))
            loss = []
            for iteration in range(int(self.max_iter)):
                p = self.sigmoid(np.dot(X, theta))
                if self.penalty == 'l2':
                    grad = -(np.dot(X.T, y_ova - p) + theta) / len(y_ova)
                elif self.penalty == 'l1':
                    l1 = np.ones_like(theta)
                    l1[np.where(theta < 0)] = 0
                    grad = -(np.dot(X.T, y_ova - p) + l1) / len(y_ova)
                
                los = 1 / len(y_ova) * np.sum(-y_ova.T.dot(np.log(p)) - (1 - y_ova).T.dot(np.log(1 - p)))
                loss.append(los)

                if (np.absolute(grad) < self.tol).all():
                    break
                theta = theta - self.lr * grad  
            self.thetas.append((theta, i))
            self.train_loss.append((loss, i))
        
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        pred = [max((self.sigmoid(np.dot(i, theta)), c) for theta, c in self.thetas)[1] for i in X]
        return pred

    def draw_loss_curve(self):
        for loss, c in self.train_loss:
            plt.subplot(2, 2, c + 1)
            plt.plot(loss)
            plt.title("Loss Curve of Class-" + str(c) +" vs All")
            plt.xlabel("Number of Iterations")
            plt.ylabel("Loss")
            plt.tight_layout()


class SVM:
    def __init__(self, C1=1e-2, C2=1e-5, lr=0.01, tol=1e-4, max_iter=1e4):
        self.thetas = []
        self.train_loss = []
        self.C1 = C1
        self.C2 = C2
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        y = np.expand_dims(y, axis=1)

        for i in np.unique(y):
            early_stop = False
            y_ova = np.where(y == i, 1, 0)
            theta = np.zeros((X.shape[1], 1))
            loss = []

            for iteration in range(int(self.max_iter)):
                z = y_ova * np.dot(X, theta)
                hinge = np.maximum(0, 1 - z)

                hinge_grad = -y_ova * X * self.C2
                hinge_grad[np.where(hinge == 0)[0]] = 0
                hinge_grad = np.sum(hinge_grad, axis=0)

                grad = hinge_grad.reshape((-1, 1)) + self.C1 * theta

                hinge_loss = np.sum(hinge) * self.C2
                los = self.C1 * 0.5 * np.linalg.norm(theta) + hinge_loss
                loss.append(los)

                if (np.absolute(grad) < self.tol).all():
                    early_stop = True
                    break
                theta = theta - self.lr * grad
            
            self.thetas.append((theta, i))
            self.train_loss.append((loss, i))

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        pred = [max((np.abs(np.dot(i, theta) + 1), c) for theta, c in self.thetas)[1] for i in X]
        return pred

    def draw_loss_curve(self):
        for loss, c in self.train_loss:
            plt.subplot(2, 2, c + 1)
            plt.plot(loss)
            plt.title("Loss Curve of Class-" + str(c) +" vs All")
            plt.xlabel("Number of Iterations")
            plt.ylabel("Loss")
            plt.tight_layout()


def dataset_drop_duplicate(dataset):
    """ 去除冗余的特征数据
    :param dataset: 目标数据集
    """
    ds = dataset.copy()
    ds = ds.drop_duplicates()
    return ds

def dataset_drop_null(dataset):
    """ 将含有空缺值的数据去除
    :param dataset: 目标数据集
    """
    ds = dataset.copy()
    ds = ds.dropna(axis=0, inplace=False)
    return ds

def dataset_replace_null(dataset):
    """ 用平均值替换空缺值
    :param dataset: 目标数据集
    """
    ds = dataset.copy()
    for col in ds.columns:
        mean_val = ds[col].mean()
        ds[col].fillna(mean_val, inplace=True)
    return ds

def dataset_drop_outlier(dataset):
    """ 用缺失值替换离群点
    :param dataset: 目标数据集
    """
    ds = dataset.copy()
    outlier = ds.boxplot(return_type='dict')
    for i, col in enumerate(ds.columns):
        x = outlier['fliers'][i].get_xdata()
        y = outlier['fliers'][i].get_ydata()
        ds[col].replace(y, np.NaN, inplace=True)
    return ds

def dataset_norm(dataset, method='min_max'):
    """ 将数据集标准化
    :param dataset: 目标数据集
    :param method: 标准化的方法
    """
    methods = ['min_max', 'z-score']
    if method is None or method not in methods:
        raise Exception("no this method.")
    ds = dataset.copy()
    if method == 'min_max':
        ds = 2 * (ds - ds.min()) / (ds.max() - ds.min()) - 1
    if method == 'z-score':
        ds = (ds - ds.mean()) / ds.std()
    return ds

def dataset_select_features(dataset, label, method='lda'):
    """ 进行特征选取
    :param dataset: 目标数据集
    :param label: 数据集的标签
    :param method: 特征选择的方法
    """
    methods = ['lasso', 'f_classif', 'mutual_info', 'random_forest', 'lda', 'relief-f']
    if method is None or method not in methods:
        raise Exception("no this method.")
    if method == 'lasso':
        ds = dataset.copy()
        lb = np.array(label.copy()).flatten()
        lasso_model = LassoCV()
        lasso_model.fit(ds, lb)
        importance = np.abs(lasso_model.coef_)
        coef = pd.Series(lasso_model.coef_, index = ds.columns)
        imp_coef = pd.concat([coef.sort_values().head(10),
                            coef.sort_values().tail(10)])
        idx = importance.argsort()[-21]
        threshold = importance[idx] + 1e-6
        sfm = SelectFromModel(lasso_model, threshold=threshold)
        sfm.fit(ds, lb)
        ds_transform = sfm.transform(ds)
        imp_coef.plot(kind="barh")
        plt.title("Coefficients in Lasso Model")
        plt.show()
        return pd.DataFrame(ds_transform)
    if method == 'f_classif':
        ds = dataset.copy()
        lb = np.array(label.copy()).flatten()
        ds_np = SelectKBest(f_classif, k=20).fit_transform(ds, lb)
        return pd.DataFrame(ds_np)
    if method == 'mutual_info':
        ds = dataset.copy()
        lb = np.array(label.copy()).flatten()
        ds_np = SelectKBest(mutual_info_classif, k=20).fit_transform(ds, lb)
        return pd.DataFrame(ds_np)
    if method == 'random_forest':
        ds = dataset.copy()
        lb = np.array(label.copy()).flatten()
        rfc_model = RandomForestClassifier(n_estimators=100)
        rfc_model.fit(ds, lb)
        importance = np.abs(rfc_model.feature_importances_)
        coef = pd.Series(rfc_model.feature_importances_, index = ds.columns)
        imp_coef = pd.concat([coef.sort_values().head(10),
                            coef.sort_values().tail(10)])
        idx = importance.argsort()[-21]
        threshold = importance[idx] + 1e-6
        sfm = SelectFromModel(rfc_model, threshold=threshold)
        sfm.fit(ds, lb)
        ds_transform = sfm.transform(ds)
        imp_coef.plot(kind="barh")
        plt.title("Coefficients in Random Forest Model")
        plt.show()
        return pd.DataFrame(ds_transform)
    if method == 'lda':
        ds = dataset.copy()
        lb = np.array(label.copy()).flatten()
        lda_model = LDA(n_components=3)
        ds_transform = lda_model.fit(ds, lb).transform(ds)
        return pd.DataFrame(ds_transform)
    if method == 'relief-f':
        ds = dataset.copy().to_numpy()
        lb = np.array(label.copy()).flatten()
        relieff_model = ReliefF(n_features_to_select=20, n_neighbors=1)
        ds_transform = relieff_model.fit_transform(ds, lb)
        return pd.DataFrame(ds_transform)


def dataset_split(dataset_X, dataset_Y, frac, random_state):
    """ 留出法随机划分数据集
    :param dataset_X: 目标数据集
    :param dataset_Y: 真实标签
    :param frac: 划分规模
    :param random_state: 随机种子
    """
    dataset = pd.concat([dataset_X, dataset_Y], axis = 1)
    dataset_train = dataset.sample(frac=frac, random_state=random_state)
    dataset_test = dataset[~dataset.index.isin(dataset_train.index)]
    X_train = np.array(dataset_train.iloc[:, :-1])
    Y_train = np.array(dataset_train.iloc[:, -1])
    X_test = np.array(dataset_test.iloc[:, :-1])
    Y_test = np.array(dataset_test.iloc[:, -1])

    return X_train, Y_train, X_test, Y_test

def holdout5_eval(estimator, X, y, model=None):
    """ 5次留出法训练并验证，并且绘制一次训练损失曲线
    :param estimator: 需要使用的算法
    :param X: 目标数据集
    :param y: 数据集标签
    :param model: 模型名称
    """
    models = ['LogisticRegression', 'DecisionTree', 'XGBoost', 'SVM', 'NeuralNetwork']
    if model is None or model not in models:
        raise Exception("no this model.")
    
    seeds = [1234, 2234, 3234, 4234, 5234]
    accuracys = []

    for seed in seeds:
        X_train, Y_train, X_test, Y_test = dataset_split(X, y, 0.8, seed)
        train_model = copy.deepcopy(estimator)
        if model == 'XGBoost':
            eval_set = [(X_train, Y_train), (X_test, Y_test)]
            train_model.fit(X_train, Y_train, eval_metric='mlogloss', eval_set=eval_set, verbose=False)
        else:
            train_model.fit(X_train, Y_train)
        y_pred = train_model.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        accuracys.append(accuracy)

    # 选择最后一个模型进行绘图
    if model == 'LogisticRegression' or model == 'SVM':
        train_model.draw_loss_curve()
    if model == 'XGBoost':
        results = train_model.evals_result()
        plt.plot(results['validation_0']['mlogloss'], label='train')
    
    print("holdout-model's average accuracy score is {}.".format(np.mean(accuracys)))
    return accuracys

def crossvalid5_eval(estimator, X, y, model=None):
    """ 5折交叉训练并验证
    :param estimator: 需要使用的算法
    :param X: 目标数据集
    :param y: 数据集标签
    :param model: 模型名称
    """
    models = ['LogisticRegression', 'DecisionTree', 'XGBoost', 'SVM', 'NeuralNetwork']
    if model is None or model not in models:
        raise Exception("no this model.")

    accuracys = []
    kf = KFold(n_splits=5, shuffle=True, random_state=3407)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_test = np.array(X_train), np.array(X_test)
        Y_train, Y_test = np.array(Y_train).flatten(), np.array(Y_test).flatten()
        train_model = copy.deepcopy(estimator)
        if model == 'XGBoost':
            eval_set = [(X_train, Y_train), (X_test, Y_test)]
            train_model.fit(X_train, Y_train, eval_metric='mlogloss', eval_set=eval_set, verbose=False)
        else:
            train_model.fit(X_train, Y_train)
        y_pred = train_model.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        accuracys.append(accuracy)

    # 选择最后一个模型进行绘图
    if model == 'LogisticRegression' or model == 'SVM':
        train_model.draw_loss_curve()
    if model == 'XGBoost':
        results = train_model.evals_result()
        plt.plot(results['validation_0']['mlogloss'], label='train')
        plt.title("Loss Curve of XGBoost")
        plt.xlabel("Number of Estimators")
        plt.ylabel("Loss")

    print("cross validation-model's average accuracy score is {}.".format(np.mean(accuracys)))
    return accuracys

def get_best_param(estimator, X, y, params, silent):
    """ 获取最佳参数
    :param estimator: 模型
    :param X: 数据集
    :param y: 标签
    :param params: 参数列表
    :param silent: 是否输出信息
    """
    y = np.array(y).flatten()
    grid_search = GridSearchCV(estimator, params, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(X, y)
    scores = grid_search.cv_results_['mean_test_score']
    if not silent:
        plt.bar(range(len(scores)), scores)
    return grid_search.best_params_