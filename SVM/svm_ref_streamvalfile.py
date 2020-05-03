import numpy as np
import joblib
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import random

path1=r'F:\bksj\jyt\TIMIT_img\train_threecharacter\LBP\attack'
path2=r'F:\bksj\jyt\TIMIT_img\train_threecharacter\LBP\real'
# path3=r'E:\lbpsvmtest\yellow'
savepath=r'F:\bksj\jyt\SVM\lbpsvm'
imglist1 = os.listdir(path1)
imglist2 = os.listdir(path2)
imglist= imglist1 + imglist2
print(imglist)
randomlist=random.shuffle(imglist)
print(imglist)
X = np.empty(shape=[0, 65536])
y = []
prefix='ori'
i=0

clf = SGDClassifier()
for list in imglist:
    if i>=100:
        clf.partial_fit(X, y, classes=np.array([0, 1]))
        i=0
        # print(X)
        # print(y)
        X = np.empty(shape=[0, 65536])
        y = []

    else:
        i += 1
        if prefix in list:
            citys = np.loadtxt(path2 + '/' + list, delimiter=' ')  # 读入txt文件，分隔符为\t
            data = np.reshape(citys, [1, 65536])  # 改变矩阵排列，统一为1*65536
            X = np.append(X, data, axis=0)
            y.append(1)
        else:
            citys = np.loadtxt(path1 + '/' + list, delimiter=' ')  # 读入txt文件，分隔符为\t
            data = np.reshape(citys, [1, 65536])  # 改变矩阵排列，统一为1*65536
            X = np.append(X, data, axis=0)
            y.append(0)

joblib.dump(clf, savepath+'/'+'clf.pkl')

