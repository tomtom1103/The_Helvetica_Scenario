# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import cvxopt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# %%
sns.set()
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.60)
plt.figure(figsize=(10,10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='rainbow');


# %%
xfit = np.linspace(-1, 3.5)
c1 = (0.5, 1.83, 0.43)
c2 = (-0.2, 2.82, 0.37)
plt.figure(figsize=(10,10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='rainbow');
for w, b, d in [c1, c2]:
    yfit = w * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, alpha=0.3)

plt.xlim(-1, 3.5);

# %%
class SVM:
    def fit(self, X, y):
        n_samples, n_features = X.shape
# P = X^T X
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = np.dot(X[i], X[j])
        P = cvxopt.matrix(np.outer(y, y) * K)
# q = -1 (1xN)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
# A = y^T 
        A = cvxopt.matrix(y, (1, n_samples))
# b = 0 
        b = cvxopt.matrix(0.0)
# -1 (NxN)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
# 0 (1xN)
        h = cvxopt.matrix(np.zeros(n_samples))
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
# Lagrange multipliers
        a = np.ravel(solution['x'])
# Lagrange have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
# Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)
# Weights
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        
    def project(self, X):
        return np.dot(X, self.w) + self.b
        
    def predict(self, X):
        return np.sign(self.project(X))

# %%
y[y == 0] = -1
tmp = np.ones(len(X))
y = tmp * y
plt.figure(figsize=(10,10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='rainbow');

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svm = SVM()
svm.fit(X_train, y_train)
def f(x, w, b, c=0):
    return (-w[0] * x - b + c) / w[1]
plt.figure(figsize=(10,10))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter')

# w.x + b = 0
a0 = -1; a1 = f(a0, svm.w, svm.b)
b0 = 4; b1 = f(b0, svm.w, svm.b)
plt.plot([a0,b0], [a1,b1], 'k')

# w.x + b = 1
a0 = -1; a1 = f(a0, svm.w, svm.b, 1)
b0 = 4; b1 = f(b0, svm.w, svm.b, 1)
plt.plot([a0,b0], [a1,b1], 'k--')
# w.x + b = -1
a0 = -1; a1 = f(a0, svm.w, svm.b, -1)
b0 = 4; b1 = f(b0, svm.w, svm.b, -1)
plt.plot([a0,b0], [a1,b1], 'k--')
y_pred = svm.predict(X_test)
confusion_matrix(y_test, y_pred)
