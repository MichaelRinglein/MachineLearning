import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
#print(iris.data)

# Splitting up the data into test and train data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0) #40% test data, 60% train data
#print(len(X_train), len(X_test)) #90 for training and 60 for testing

# Building SVC model for predicting the iris classifications
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train) #train data
result = clf.score(X_test, y_test) #test data
print('clf.score is: ', result)

# K-Fold Cross Validation
scores = cross_val_score(clf, iris.data, iris.target, cv=5) # (estimator, X, y, number of folds)
print('scores are: ', scores)
print('scores.mean are: ', scores.mean())

# Trying poly instead of linear for the kernel in clf
clf = svm.SVC(kernel='poly', C=1).fit(X_train, y_train)
result = clf.score(X_test, y_test)
print('result poly is: ', result)

# Are we maybe overfitting?
# Let's test it via K-Fold Cross Validation

scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print('scores poly are:', scores)
print('scores.mean poly are:', scores.mean()) 

# The result with a poly kernel is worse than with a linear kernel
# This is a clear sign that we overfit the model with a poly kernel

# Trying a 2-degree poly kernel
clf = svm.SVC(kernel='poly', degree=2, C=1).fit(X_train, y_train)
result = clf.score(X_test, y_test)
print('result poly 2 degree is: ', result)

scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print('scores poly 2 degree is: ', scores)
print('scores.mean poly 2 degree is: ', scores.mean())

# With 2 dimensions, the poly kernel is better than with 3 dimensions
# But still not as good as the linear kernel

# The linear kernel seems to be the best score with the least error




