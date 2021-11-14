import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as matplt


def pca_function(X, no_components):
    X_meaned = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_meaned, rowvar=False)

    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    eigen_vector_subset = sorted_eigenvectors[:, 0:no_components]

    X_reduced = np.dot(eigen_vector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Target'])

x = data.iloc[:, 0:4]

target = data.iloc[:, 4]
mat_reduced = pca_function(x, 2)

final_df = pd.DataFrame(mat_reduced, columns=['Eigen Vectors', 'Mean Centered Data'])
final_df = pd.concat([final_df, pd.DataFrame(target)], axis=1)

matplt.figure(figsize=(6, 6))
sb.scatterplot(data=final_df, x='Eigen Vectors', y='Mean Centered Data', hue='Target', s=60, palette='hls')
matplt.show()
