from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle
from pylab import *

# Importing iris data from scikit-learn
iris = load_iris()
numSamples, numFeatures = iris.data.shape
print(numSamples)
print(numFeatures) #4 Dimensions, Petal length and width and Sepal length and width
print(list(iris.target_names)) #3 different kind of iris

# Changing it to 2 dimension
x = iris.data
pca = PCA(n_components=2, whiten=True).fit(x)
X_pca = pca.transform(x)
print(pca.components_)

# Checking how much of the original variance is preserved after 4d to 2d transform
print(pca.explained_variance_ratio_) #92% in first dimension + 5% in second dimension preserved
print(sum(pca.explained_variance_ratio_)) #together less than 3% variance is lost

# Plotting it
colors = cycle('rgb')
target_ids = range(len(iris.target_names))
pl.figure()
for i, c, label in zip(target_ids, colors, iris.target_names):
	pl.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1],
	c=c, label=label)
pl.legend()
pl.title('2-dimensional representation of 4-dimensional data')
pl.show()	
	
