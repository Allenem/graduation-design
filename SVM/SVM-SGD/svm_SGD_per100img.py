import os
import joblib
import datetime
import numpy as np
from sklearn.linear_model import SGDClassifier
from GetData.get_color import get_color
from GetData.get_SURF import get_SURF
from GetData.get_ELA import get_ELA


inputpath_train = 'G:/SVM/Train'
inputpath_test = 'G:/SVM/Test'
savepath = './Model/'
train_true = inputpath_train+'/True'
train_fake = inputpath_train+'/Fake'
test_true = inputpath_test+'/True'
test_fake = inputpath_test+'/Fake'
train_true_lists = os.listdir(train_true)
train_fake_lists = os.listdir(train_fake)
test_true_lists = os.listdir(test_true)
test_fake_lists = os.listdir(test_fake)
color_true_data_train = []
color_fake_data_train = []
SURF_true_data_train = []
SURF_fake_data_train = []
ELA_true_data_train = []
ELA_fake_data_train = []
color_true_data_test = []
color_fake_data_test = []
SURF_true_data_test = []
SURF_fake_data_test = []
ELA_true_data_test = []
ELA_fake_data_test = []
X = []
Y = []
accuracy_set = []


def train_color():
    print('color特征SVM分类器训练开始 ……')
    color_clf = SGDClassifier()
    excel_num = 0
    for train_true_list in train_true_lists:
        if train_true_list.endswith('color.xlsx'):
            inputpath = train_true + '/' + train_true_list
            sheets, sheets_1dim = get_color(inputpath)
            if excel_num % 10 == 0:
                color_true_data_train = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    color_true_data_train.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1
            if excel_num % 10 == 0:
                X = color_true_data_train
                color_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                X = []
                Y = []

    excel_num = 0
    for train_fake_list in train_fake_lists:
        if train_fake_list.endswith('color.xlsx'):
            inputpath = train_fake + '/' + train_fake_list
            sheets, sheets_1dim = get_color(inputpath)
            if excel_num % 10 == 0:
                color_fake_data_train = sheets_1dim
                Y = [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    color_fake_data_train.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1
            if excel_num % 10 == 0:
                X = color_fake_data_train
                color_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                X = []
                Y = []

    joblib.dump(color_clf, savepath + 'color_clf.pkl')
    excel_num = 0
    X = []
    Y = []


def test_color():
    print('color特征SVM分类器测试开始 ……')
    color_clf2 = joblib.load(savepath + 'color_clf.pkl')
    excel_num = 0
    for test_true_list in test_true_lists:
        if test_true_list.endswith('color.xlsx'):
            inputpath = test_true + '/' + test_true_list
            sheets, sheets_1dim = get_color(inputpath)
            if excel_num % 10 == 0:
                color_true_data_test = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    color_true_data_test.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1
            if excel_num % 10 == 0:
                X = color_true_data_test
                Z = color_clf2.predict(X)
                accuracy = color_clf2.score(X, Y)
                accuracy_set.append(accuracy)
                X = []
                Y = []
                Z = []

    excel_num = 0
    for test_fake_list in test_fake_lists:
        if test_fake_list.endswith('color.xlsx'):
            inputpath = test_fake + '/' + test_fake_list
            sheets, sheets_1dim = get_color(inputpath)
            if excel_num % 10 == 0:
                color_fake_data_test = sheets_1dim
                Y = [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    color_fake_data_test.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1
            if excel_num % 10 == 0:
                X = color_fake_data_test
                Z = color_clf2.predict(X)
                accuracy = color_clf2.score(X, Y)
                accuracy_set.append(accuracy)
                X = []
                Y = []
                Z = []

    print('color_clf average accuracy: {}'.format(
        sum(accuracy_set)/len(accuracy_set)))
    excel_num = 0
    X = []
    Y = []
    Z = []
    accuracy_set = []


def train_SURF():
    print('SURF特征SVM分类器训练开始 ……')
    SURF_clf = SGDClassifier()
    excel_num = 0
    for train_true_list in train_true_lists:
        if train_true_list.endswith('SURF.xlsx'):
            inputpath = train_true + '/' + train_true_list
            sheets, sheets_1dim = get_SURF(inputpath)
            if excel_num % 10 == 0:
                SURF_true_data_train = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    SURF_true_data_train.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1
            if excel_num % 10 == 0:
                X = SURF_true_data_train
                SURF_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                X = []
                Y = []

    excel_num = 0
    for train_fake_list in train_fake_lists:
        if train_fake_list.endswith('SURF.xlsx'):
            inputpath = train_fake + '/' + train_fake_list
            sheets, sheets_1dim = get_SURF(inputpath)
            if excel_num % 10 == 0:
                SURF_fake_data_train = sheets_1dim
                Y = [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    SURF_fake_data_train.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1
            if excel_num % 10 == 0:
                X = SURF_fake_data_train
                SURF_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                X = []
                Y = []

    joblib.dump(SURF_clf, savepath + 'SURF_clf.pkl')
    excel_num = 0
    X = []
    Y = []


def test_SURF():
    print('SURF特征SVM分类器测试开始 ……')
    SURF_clf2 = joblib.load(savepath + 'SURF_clf.pkl')
    excel_num = 0
    for test_true_list in test_true_lists:
        if test_true_list.endswith('SURF.xlsx'):
            inputpath = test_true + '/' + test_true_list
            sheets, sheets_1dim = get_SURF(inputpath)
            if excel_num % 10 == 0:
                SURF_true_data_test = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    SURF_true_data_test.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1
            if excel_num % 10 == 0:
                X = SURF_true_data_test
                Z = SURF_clf2.predict(X)
                accuracy = SURF_clf2.score(X, Y)
                accuracy_set.append(accuracy)
                X = []
                Y = []
                Z = []

    excel_num = 0
    for test_fake_list in test_fake_lists:
        if test_fake_list.endswith('SURF.xlsx'):
            inputpath = test_fake + '/' + test_fake_list
            sheets, sheets_1dim = get_SURF(inputpath)
            if excel_num % 10 == 0:
                SURF_fake_data_test = sheets_1dim
                Y = [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    SURF_fake_data_test.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1
            if excel_num % 10 == 0:
                X = SURF_fake_data_test
                Z = SURF_clf2.predict(X)
                accuracy = SURF_clf2.score(X, Y)
                accuracy_set.append(accuracy)
                X = []
                Y = []
                Z = []

    print('SURF_clf average accuracy: {}'.format(
        sum(accuracy_set)/len(accuracy_set)))
    excel_num = 0
    X = []
    Y = []
    Z = []
    accuracy_set = []


def train_ELA():
    print('ELA特征SVM分类器训练开始 ……')
    ELA_clf = SGDClassifier()
    excel_num = 0
    for train_true_list in train_true_lists:
        if train_true_list.endswith('ELA.xlsx'):
            inputpath = train_true + '/' + train_true_list
            sheets, sheets_1dim = get_ELA(inputpath)
            if excel_num % 10 == 0:
                ELA_true_data_train = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    ELA_true_data_train.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1
            if excel_num % 10 == 0:
                X = ELA_true_data_train
                ELA_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                X = []
                Y = []

    excel_num = 0
    for train_fake_list in train_fake_lists:
        if train_fake_list.endswith('ELA.xlsx'):
            inputpath = train_fake + '/' + train_fake_list
            sheets, sheets_1dim = get_ELA(inputpath)
            if excel_num % 10 == 0:
                ELA_fake_data_train = sheets_1dim
                Y = [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    ELA_fake_data_train.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1
            if excel_num % 10 == 0:
                X = ELA_fake_data_train
                ELA_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                X = []
                Y = []

    joblib.dump(ELA_clf, savepath + 'ELA_clf.pkl')
    excel_num = 0
    X = []
    Y = []


def test_ELA():
    print('ELA特征SVM分类器测试开始 ……')
    ELA_clf2 = joblib.load(savepath + 'ELA_clf.pkl')
    excel_num = 0
    for test_true_list in test_true_lists:
        if test_true_list.endswith('ELA.xlsx'):
            inputpath = test_true + '/' + test_true_list
            sheets, sheets_1dim = get_ELA(inputpath)
            if excel_num % 10 == 0:
                ELA_true_data_test = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    ELA_true_data_test.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1
            if excel_num % 10 == 0:
                X = ELA_true_data_test
                Z = ELA_clf2.predict(X)
                accuracy = ELA_clf2.score(X, Y)
                accuracy_set.append(accuracy)
                X = []
                Y = []
                Z = []

    excel_num = 0
    for test_fake_list in test_fake_lists:
        if test_fake_list.endswith('ELA.xlsx'):
            inputpath = test_fake + '/' + test_fake_list
            sheets, sheets_1dim = get_ELA(inputpath)
            if excel_num % 10 == 0:
                ELA_fake_data_test = sheets_1dim
                Y = [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    ELA_fake_data_test.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1
            if excel_num % 10 == 0:
                X = ELA_fake_data_test
                Z = ELA_clf2.predict(X)
                accuracy = ELA_clf2.score(X, Y)
                accuracy_set.append(accuracy)
                X = []
                Y = []
                Z = []

    print('ELA_clf average accuracy: {}'.format(
        sum(accuracy_set)/len(accuracy_set)))
    excel_num = 0
    X = []
    Y = []
    Z = []
    accuracy_set = []


def print_runtime(function, string):
    start = datetime.datetime.now()
    function()
    end = datetime.datetime.now()
    print('Running Time of {} : {}\n'.format(string, (end - start)))


if __name__ == '__main__':

    print_runtime(train_color, '训练color特征SVM分类器')
    print_runtime(test_color, '测试color特征SVM分类器')
    print_runtime(train_SURF, '训练SURF特征SVM分类器')
    print_runtime(test_SURF, '测试SURF特征SVM分类器')
    print_runtime(train_ELA, '训练ELA特征SVM分类器')
    print_runtime(test_ELA, '测试ELA特征SVM分类器')
