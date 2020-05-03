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
savepath = 'G:/SVM/Model'
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


def train_color():
    print('color特征SVM分类器训练开始 ……')
    excel_num = 0
    for train_true_list in train_true_lists:
        if train_true_list.endswith('color.xlsx'):
            inputpath = train_true + '/' + train_true_list
            sheets, sheets_1dim = get_color(inputpath)
            if excel_num == 0:
                color_true_data_train = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    color_true_data_train.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1

    excel_num = 0
    for train_fake_list in train_fake_lists:
        if train_fake_list.endswith('color.xlsx'):
            inputpath = train_fake + '/' + train_fake_list
            sheets, sheets_1dim = get_color(inputpath)
            if excel_num == 0:
                color_fake_data_train = sheets_1dim
                Y += [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    color_fake_data_train.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1
    print('训练数据实际真假：{}'.format(Y))

    X = color_true_data_train + color_fake_data_train
    color_clf = SGDClassifier()
    color_clf.partial_fit(X, Y, classes=np.array([0, 1]))
    joblib.dump(color_clf, savepath + '/' + 'color_clf.pkl')

    excel_num = 0
    X = []
    Y = []


def test_color():
    print('color特征SVM分类器测试开始 ……')
    excel_num = 0
    for test_true_list in test_true_lists:
        if test_true_list.endswith('color.xlsx'):
            inputpath = test_true + '/' + test_true_list
            sheets, sheets_1dim = get_color(inputpath)
            if excel_num == 0:
                color_true_data_test = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    color_true_data_test.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1

    excel_num = 0
    for test_fake_list in test_fake_lists:
        if test_fake_list.endswith('color.xlsx'):
            inputpath = test_fake + '/' + test_fake_list
            sheets, sheets_1dim = get_color(inputpath)
            if excel_num == 0:
                color_fake_data_test = sheets_1dim
                Y += [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    color_fake_data_test.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1

    X = color_true_data_test + color_fake_data_test
    color_clf2 = joblib.load(savepath+'/'+'color_clf.pkl')
    Z = color_clf2.predict(X)
    accuracy = color_clf2.score(X, Y)
    print('测试数据实际真假：{}'.format(Y))
    print('测试数据预测真假：{}'.format(Z))
    print('预测准确率：{}'.format(accuracy))

    excel_num = 0
    X = []
    Y = []


def train_SURF():
    print('SURF特征SVM分类器训练开始 ……')
    excel_num = 0
    for train_true_list in train_true_lists:
        if train_true_list.endswith('SURF.xlsx'):
            inputpath = train_true + '/' + train_true_list
            sheets, sheets_1dim = get_SURF(inputpath)
            if excel_num == 0:
                SURF_true_data_train = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    SURF_true_data_train.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1

    excel_num = 0
    for train_fake_list in train_fake_lists:
        if train_fake_list.endswith('SURF.xlsx'):
            inputpath = train_fake + '/' + train_fake_list
            sheets, sheets_1dim = get_SURF(inputpath)
            if excel_num == 0:
                SURF_fake_data_train = sheets_1dim
                Y += [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    SURF_fake_data_train.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1
    print('训练数据实际真假：{}'.format(Y))

    X = SURF_true_data_train + SURF_fake_data_train
    # print(type(X))  # list
    # print(len(X))
    # print(len(Y))

    SURF_clf = SGDClassifier()
    SURF_clf.partial_fit(X, Y, classes=np.array([0, 1]))
    joblib.dump(SURF_clf, savepath + '/' + 'SURF_clf.pkl')

    excel_num = 0
    X = []
    Y = []


def test_SURF():
    print('SURF特征SVM分类器测试开始 ……')
    excel_num = 0
    for test_true_list in test_true_lists:
        if test_true_list.endswith('SURF.xlsx'):
            inputpath = test_true + '/' + test_true_list
            sheets, sheets_1dim = get_SURF(inputpath)
            if excel_num == 0:
                SURF_true_data_test = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    SURF_true_data_test.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1

    excel_num = 0
    for test_fake_list in test_fake_lists:
        if test_fake_list.endswith('SURF.xlsx'):
            inputpath = test_fake + '/' + test_fake_list
            sheets, sheets_1dim = get_SURF(inputpath)
            if excel_num == 0:
                SURF_fake_data_test = sheets_1dim
                Y += [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    SURF_fake_data_test.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1

    X = SURF_true_data_test + SURF_fake_data_test
    SURF_clf2 = joblib.load(savepath+'/'+'SURF_clf.pkl')
    Z = SURF_clf2.predict(X)
    accuracy = SURF_clf2.score(X, Y)
    print('测试数据实际真假：{}'.format(Y))
    print('测试数据预测真假：{}'.format(Z))
    print('预测准确率：{}'.format(accuracy))

    excel_num = 0
    X = []
    Y = []


def train_ELA():
    print('ELA特征SVM分类器训练开始 ……')
    excel_num = 0
    for train_true_list in train_true_lists:
        if train_true_list.endswith('ELA.xlsx'):
            inputpath = train_true + '/' + train_true_list
            sheets, sheets_1dim = get_ELA(inputpath)
            if excel_num == 0:
                ELA_true_data_train = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    ELA_true_data_train.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1

    excel_num = 0
    for train_fake_list in train_fake_lists:
        if train_fake_list.endswith('ELA.xlsx'):
            inputpath = train_fake + '/' + train_fake_list
            sheets, sheets_1dim = get_ELA(inputpath)
            if excel_num == 0:
                ELA_fake_data_train = sheets_1dim
                Y += [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    ELA_fake_data_train.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1
    print('训练数据实际真假：{}'.format(Y))

    X = ELA_true_data_train + ELA_fake_data_train
    ELA_clf = SGDClassifier()
    ELA_clf.partial_fit(X, Y, classes=np.array([0, 1]))
    joblib.dump(ELA_clf, savepath + '/' + 'ELA_clf.pkl')

    excel_num = 0
    X = []
    Y = []


def test_ELA():
    print('ELA特征SVM分类器测试开始 ……')
    excel_num = 0
    for test_true_list in test_true_lists:
        if test_true_list.endswith('ELA.xlsx'):
            inputpath = test_true + '/' + test_true_list
            sheets, sheets_1dim = get_ELA(inputpath)
            if excel_num == 0:
                ELA_true_data_test = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    ELA_true_data_test.append(sheets_1dim[i])
                    Y.append(1)
            excel_num += 1

    excel_num = 0
    for test_fake_list in test_fake_lists:
        if test_fake_list.endswith('ELA.xlsx'):
            inputpath = test_fake + '/' + test_fake_list
            sheets, sheets_1dim = get_ELA(inputpath)
            if excel_num == 0:
                ELA_fake_data_test = sheets_1dim
                Y += [0] * len(sheets_1dim)
            else:
                for i in range(len(sheets_1dim)):
                    ELA_fake_data_test.append(sheets_1dim[i])
                    Y.append(0)
            excel_num += 1

    X = ELA_true_data_test + ELA_fake_data_test
    ELA_clf2 = joblib.load(savepath+'/'+'ELA_clf.pkl')
    Z = ELA_clf2.predict(X)
    accuracy = ELA_clf2.score(X, Y)
    print('测试数据实际真假：{}'.format(Y))
    print('测试数据预测真假：{}'.format(Z))
    print('预测准确率：{}'.format(accuracy))

    excel_num = 0
    X = []
    Y = []


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

    # train_color()
    # test_color()
    # train_SURF()
    # test_SURF()
    # train_ELA()
    # test_ELA()

    # 测试get_color,get_SURF,get_ELA用代码
    # sheets, sheets_1dim = get_ELA(
    #     'G:/SVM/Train/True/celeba_devel_ELA.xlsx')
    # print(len(sheets_1dim))
    # print(len(sheets_1dim[0]))
    # print(type(sheets_1dim))
