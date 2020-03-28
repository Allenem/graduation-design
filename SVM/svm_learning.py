from sklearn import svm

# training samples
X = [[0, 0], [0, 1], [1, 0,], [1, 1]]
# training targets
Y = [0, 1, 2, 3]
# class
# Parameters: 
# C: float, optional (default=1.0)
# kernel: string, optional (default=’rbf’)
# degree: int, optional (default=3)
# gamma: {‘scale’, ‘auto’} or float, optional (default=’scale’)
# ...
clf = svm.SVC()
# Fit the SVM model according to the given training data.
clf.fit(X, Y)
# Return the mean accuracy on the given test data and labels.
accuracy_result = clf.score([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 3])
print(accuracy_result)
# Perform classification on samples in X.
predict_result = clf.predict([[0, 1]])
print(predict_result)

# OUTPUT
# 0.75
# [1]