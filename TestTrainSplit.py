import numpy as np
import matplotlib.pyplot as plt
from pylab import *
np.random.seed(2)

# Creating data
pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds
#plt.scatter(pageSpeeds, purchaseAmount)
#plt.show()

# Split into Training and Test data
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]
trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]
plt.scatter(trainX, trainY, c='b')
plt.scatter(testX, testY, c='r')
plt.show() #blue ist training data and red is test data

# Trying to fit a 8th degree polynominal function to the data
x = np.array(trainX)
y = np.array(trainY)
p8 = np.poly1d(np.polyfit(x, y, 8))

# Plotting the polynominal function against training  data
xp = np.linspace(0, 7, 100) #from 0 to 7 seconds page load time
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
plt.title('Training data & 8th degree polynominal function')
plt.scatter(x, y)
plt.plot(xp, p8(xp), c='r')
plt.show() #8 degrees are overfitting after 5 seconds page load

# Plotting the polynominal function against test data
testx = np.array(testX)
testy = np.array(testY)
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])
plt.title('Test data & 8th degree polynominal function')
plt.scatter(testx, testy)
plt.plot(xp, p8(xp), c='r')
plt.show()

# Checking the r-squared
from sklearn.metrics import r2_score
r2 = r2_score(testy, p8(testx))
print('r-squared error for the test data is: ', r2) #0.30018 - bad score, functions fits bad to the data
trainingx = np.array(trainX)
trainingy = np.array(trainY)
r2_training = r2_score(trainingy, p8(trainingx))
print('r-squared error for the training data is: ', r2_training) #0.64270 - better fit to training data

# So let's try polynominal functions with less degrees
p4 = np.poly1d(np.polyfit(x, y, 4)) #testing 4th degree polynominal function
r2_4th_test = r2_score(testy, p4(testx))
print('r2_4th_test is: ', r2_4th_test)
r2_4th_training = r2_score(trainingy, p4(trainingx))
print('r2_4th_training is:', r2_4th_training)  

p3 = np.poly1d(np.polyfit(x, y, 3)) # testing 3rd degree polynominal  function
r2_3rd_test = r2_score(testy, p3(testx))
print('r2_3rd_test is: ', r2_3rd_test)
r2_3rd_training = r2_score(trainingy, p3(trainingx))
print('r2_3rd_training is: ', r2_3rd_training)

p2 = np.poly1d(np.polyfit(x, y, 2)) # testing 2nd degree polynominal function
r2_2nd_test = r2_score(testy, p2(testx))
print('r2_2nd_test is: ', r2_2nd_test)
r2_2nd_training = r2_score(trainingy, p2(trainingx))
print('r2_2nd_training is: ', r2_2nd_training)

# Summary: No matter how many degrees, the function never fits the data well





