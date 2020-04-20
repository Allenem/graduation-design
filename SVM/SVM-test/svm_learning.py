from sklearn import svm
import pickle

# training samples
X = [[0, 0], [0, 1], [1, 0,], [1, 1]]
# training targets
Y = [0, 1, 2, 3]
# class
# Parameters: 
# C: float, optional (default = 1.0)
# kernel: string, optional (default = 'rbf')
# degree: int, optional (default = 3)
# gamma: {'scale', 'auto'} or float, optional (default = 'scale')
# ...
clf = svm.SVC()
# Fit the SVM model according to the given training data.
clf.fit(X, Y)
# Return the mean accuracy on the given test data and labels.
accuracy_result = clf.score([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 3])
print(accuracy_result)  # 0.75
# Perform classification on samples in X.
predict_result = clf.predict([[0, 1]])
print(predict_result)  # [1]

# write model into .pickle file and reuse it
modelfile = open("./model.pickle", "wb")
pickle.dump(clf, modelfile)
modelfile.close()
modelfile = open("./model.pickle", "rb")
svm_model = pickle.load(modelfile)
print(svm_model)
modelfile.close()
pred = svm_model.predict([[0, 1], [0, 0]])
print(pred)  # [1 0]


# OUTPUT
# 0.75
# [1]
# SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# [1 0]