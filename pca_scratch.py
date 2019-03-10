import pandas as pd
df = pd.read_csv(filepath_or_buffer="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, sep=",")
df.head()

df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.describe()

df.dropna(how='all', inplace=True)

df.tail()

X = df.iloc[:,0:4].values
y = df.iloc[:,4].values


######### Step 1- Standardise the data : Putting the data on smae scale unit:

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

######### Step 2- wiill build eigen decomposition on Covariance matrix:

# eigen decomposition means will get pair of eigen values and eigen vectors
# Covariance marix describes data and covariance among variables
# Covariance means how to variable changes w.r.t each other If + ve then positive covariance and vice versa.
# so, by implmenting eigen decomposition on covarinace matrix helps to find hidden forces on data

# so will first calculate covariance matrix:
import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

# Another way to obtain covariance matrix is using numpy cov function:
cov_mat = np.cov(X_std.T)

# Eigen Decomposition on Covariance matrix:
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# Instead of doing eigen decomposition on covariance matrix :
# We can also use SVD also as most PCA's use for computational easiness:
u,s,v = np.linalg.svd(X_std.T)
u

# step - 3 : select principle components using eigen_pairs

# We want to select principle componants:
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

# So for sort
eig_pairs.sort(key = lambda x: x[0], reverse=True)

for i in eig_pairs:
    print(i[0])

# After sorting eigen pairs next this is how many principal componants to choose.
# So will cal expalined variance : which obtaine from eigen values:
# and tells us how much variance can be attributed to each principle componants

tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in eig_vals]
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(4), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(4), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

# above graph clearly shows that first two principal componants aroung 96% of information. so we can easily drop others.

# Step - 4: Projection Matrix :
# PRojecttion Matrix to transform data on new feature subspace:
# it's is matrix which have concatenated top k eigen vectors.

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)

#  Step - 5 : Now, will use this projection matrix to transform to our new subplace via simple dot product operations:
Y = X_std.dot(matrix_w)


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
        plt.scatter(Y[y==lab, 0], Y[y==lab, 1], label=lab, c=col)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()




########################## Using sklearn: Short Cut ################################

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
        plt.scatter(Y_sklearn[y==lab, 0], Y_sklearn[y==lab, 1], label=lab, c=col)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()