Dimensionality Reduction Lab: PCA, LDA, and KPCA
Overview

This lab explores dimensionality reduction techniques in machine learning:

PCA (Principal Component Analysis) – unsupervised, maximizes variance.

LDA (Linear Discriminant Analysis) – supervised, maximizes class separability.

KPCA (Kernel PCA) – nonlinear data using RBF kernel.

Datasets used:

Wine dataset (for PCA and LDA)

Synthetic half-moon and circles datasets (for KPCA)

Classification performance is evaluated using Logistic Regression.

Prerequisites

Python 3.x

Libraries: numpy, pandas, matplotlib, scikit-learn, scipy

Jupyter Notebook recommended

Install required packages:

pip install numpy pandas matplotlib scikit-learn scipy

Setup

Clone the repository:

git clone <repo-url>
cd lab-dimensionality-reduction


Launch Jupyter Notebook:

jupyter notebook


Open dimensionality_reduction_lab.ipynb.

Download the Wine dataset if needed:

https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

Key Code Snippets
1. PCA with scikit-learn
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

2. LDA with scikit-learn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

3. Custom RBF Kernel PCA
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

def rbf_kernel_pca(X, gamma, n_components):
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = np.exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K_centered)
    X_pc = np.column_stack([eigvecs[:, -i] for i in range(1, n_components + 1)])
    return X_pc

Analysis and Observations
Explained Variance

Cumulative variance increases with the number of PCA components.

For the Wine dataset, 2–3 components explain ~95% of the variance.

PCA vs. LDA

PCA captures variance without labels; LDA uses class information.

LDA projections show better class separability, improving classifier performance.

KPCA Gamma Parameter

Small γ (e.g., 0.01): insufficient separation.

Large γ (e.g., 100): overfitting and distorted transformation.

Optimal γ produces linear separability in the transformed space.

Classifier Performance
Dataset	Original	PCA	LDA
Wine	Moderate	Improved	Best

Dimensionality reduction reduces computation time and improves accuracy for classifiers like Logistic Regression or SVM.

Limitations

PCA fails on nonlinearly separable data (e.g., half-moon, circles).

KPCA maps data into higher-dimensional space to capture nonlinear patterns, allowing linear separation.
