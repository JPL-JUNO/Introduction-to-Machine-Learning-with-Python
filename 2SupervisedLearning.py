import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
import warnings
warnings.filterwarnings('ignore')


print('--' * 15 + '2.1分类与回归' + '--' * 15)
print('--' * 15 + '2.2泛化、过拟合与欠拟合' + '--' * 15)
print('--' * 15 + '2.3监督学习算法' + '--' * 15)
print('--' * 15 + '2.3.1一些样本数据集' + '--' * 15)

X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['class 0', 'class 1'], loc=4)
plt.xlabel('First feature')
plt.ylabel('Second feature')
print('X.shape: {}'.format(X.shape))
# plt.show()


X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('Feature')
plt.ylabel('Target')
# plt.show()

cancer = load_breast_cancer()
print('cancer.keys(): \n{}'.format(cancer.keys()))

print('Shape of cancer data: {}'.format(cancer.data.shape))

print('Sample counts per class: \n{}'.format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

print('Feature names: \n{}'.format(cancer.feature_names))

boston = load_boston()
print('Data shape: {}'.format(boston.data.shape))

X, y = mglearn.datasets.load_extended_boston()
print('X.shape: {}'.format(X.shape))

print('--' * 15 + '2.3.2k近邻' + '--' * 15)

mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()
