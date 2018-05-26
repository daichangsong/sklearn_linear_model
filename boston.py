import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
import warnings

warnings.filterwarnings('ignore')


# mse
def mse(y_train, y_train_pred):
    # mse
    return sum((y_train_pred - y_train) ** 2) / y_train.shape[0]
    # rmse
    # return np.sqrt(sum((y_train_pred-y_train)**2)/y_train.shape[0])


def run_main():

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 获取数据集
    boston = load_boston()
    # 属性列
    X = boston.data
    # 标签列
    y = boston.target
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)

    print('样本个数：', X.shape[0], ' 样本维度：', X.shape[1])

    # 线性回归模型
    # ridge 回归
    ridgecv = linear_model.RidgeCV(cv=5)
    # 训练回归模型
    ridgecv.fit(x_train, y_train)
    # 获取训练集和测试集上的预测结果
    ridge_train_pred = ridgecv.predict(x_train)
    ridge_test_pred = ridgecv.predict(x_test)
    print('ridge非零特征个数：', np.sum(ridgecv.coef_ != 0))

    lassocv = linear_model.LassoCV(cv=5)
    lassocv.fit(x_train, y_train)
    lasso_train_pred = lassocv.predict(x_train)
    lasso_test_pred = lassocv.predict(x_test)
    print('lasso非零特征个数：', np.sum(lassocv.coef_ != 0))

    elasticnetcv = linear_model.ElasticNetCV(cv=5)
    elasticnetcv.fit(x_train, y_train)
    elasticnetcv_train_pred = elasticnetcv.predict(x_train)
    elasticnetcv_test_pred = elasticnetcv.predict(x_test)
    print('elasticnetcv非零特征个数：', np.sum(elasticnetcv.coef_ != 0))

    # 计算训练集和测试集上mse
    print('ridge回归训练集mse：', metrics.mean_squared_error(y_train, ridge_train_pred))
    print('ridge回归测试集mse：', metrics.mean_squared_error(y_test, ridge_test_pred))
    print('lasso回归训练集mse：', metrics.mean_squared_error(y_train, lasso_train_pred))
    print('lasso回归测试集mse：', metrics.mean_squared_error(y_test, lasso_test_pred))
    print('elasticnetcv回归训练集mse：', metrics.mean_squared_error(y_train, elasticnetcv_train_pred))
    print('elasticnetcv回归测试集mse：', metrics.mean_squared_error(y_test, elasticnetcv_test_pred))

    # plt.figure(facecolor='w')
    # plt.plot(y_test, 'r', ridge_test_pred, 'g', linewidth=2)
    # plt.xlabel('X', fontsize=15)
    # plt.ylabel('Y', fontsize=15)
    # plt.grid(True)
    # plt.show()


if __name__ == '__main__':
    run_main()
