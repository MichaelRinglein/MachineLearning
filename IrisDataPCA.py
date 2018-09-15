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
X = iris.data
pca = PCA(n_components=2, whiten=True).fit(X)
X_pca = pca.transform(X)
#print(pca.components_)

# Checking how much of the original variance is preserved after 4d to 2d transform
#print(pca.explained_variance_ratio_) #92% in first dimension + 5% in second dimension preserved
print('Variance ratio for the 2-dimensional data is: ', sum(pca.explained_variance_ratio_)) #together less than 3% variance is lost

# Plotting the 2 dimensional 
colors = cycle('rgb')
target_ids = range(len(iris.target_names))
pl.figure()
for i, c, label in zip(target_ids, colors, iris.target_names):
	pl.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1],
	c=c, label=label)
pl.legend()
pl.title('2-dimensional representation of 4-dimensional data')
pl.show() #As can be seen, it is still possible to do a classification with the 2-dimensional data

# Trying to distill it down into one dimension only
pca_1 = PCA(n_components=1, whiten=True).fit(X)
X_pca_1 = pca_1.transform(X)
#print(pca_1.components_)
print('Variance ratio for the 1-dimensional data is: ', sum(pca_1.explained_variance_ratio_)) #Even with just 1 dimension, still 92% of the data preserved!

#Explanation for this could be that the petal and sepal lengths and widths are most likely in a relation to eachother
#Therefore the dimensions can easily be reduced without loosing much variance


	
