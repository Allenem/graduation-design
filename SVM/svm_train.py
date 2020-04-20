import os
import cv2
from sklearn import svm

trainset = []
target = ['真脸', '假脸']

def getX(inputpath):
    image = cv2.imread(inputpath)
    channels = cv2.split(image)
    colors = ('b', 'g', 'r')
    # print(channels)
    with open('log.txt', "a", encoding="utf-8") as fi:
        fi.write(str(channels))
    # trainset = channels 

def train_clf():
    clf = svm.SVC()
    clf.fit(trainset, target)

def predict(testset):
    predict_result = clf.predict(testset)
    print(predict_result)

if __name__ == '__main__':
    inputfolder = 'G:/Feature_test/test/devel'
    files = os.listdir(inputfolder)
    for fil in files:
        getX(inputfolder + '/' + fil)