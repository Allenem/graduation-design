import os
import joblib
import datetime
import numpy as np
from sklearn.linear_model import SGDClassifier
from GetData.get_color import get_color
from GetData.get_SURF import get_SURF
from GetData.get_ELA import get_ELA


inputpath_train = 'G:/SVM/Z_Train'
inputpath_test = 'G:/SVM/Z_Test'
savepath = 'D:/1puyao/engineer/0github/graduation-design/SVM/SVM-SGD/Model/'
# savepath = 'G:/SVM/Model/'
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
                for k in sheets_1dim:
                    color_true_data_train.append(k)
                    Y.append(1)
            excel_num += 1
            if excel_num % 10 == 0:
                X = color_true_data_train
                color_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                joblib.dump(color_clf, savepath + 'color_clf.pkl')
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
                for k in sheets_1dim:
                    color_fake_data_train.append(k)
                    Y.append(0)
            excel_num += 1
            if excel_num % 10 == 0:
                X = color_fake_data_train
                color_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                joblib.dump(color_clf, savepath + 'color_clf.pkl')
                X = []
                Y = []

    # joblib.dump(color_clf, savepath + 'color_clf.pkl')
    excel_num = 0
    X = []
    Y = []


def test_color():
    print('color特征SVM分类器测试开始 ……')
    color_clf2 = joblib.load(savepath + 'color_clf.pkl')
    excel_num = 0
    global accuracy_set
    for test_true_list in test_true_lists:
        if test_true_list.endswith('color.xlsx'):
            inputpath = test_true + '/' + test_true_list
            sheets, sheets_1dim = get_color(inputpath)
            if excel_num % 10 == 0:
                color_true_data_test = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for k in sheets_1dim:
                    color_true_data_test.append(k)
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
                for k in sheets_1dim:
                    color_fake_data_test.append(k)
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
                for k in sheets_1dim:
                    SURF_true_data_train.append(k)
                    Y.append(1)
            excel_num += 1
            if excel_num % 10 == 0:
                X = SURF_true_data_train
                SURF_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                joblib.dump(SURF_clf, savepath + 'SURF_clf.pkl')
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
                for k in sheets_1dim:
                    SURF_fake_data_train.append(k)
                    Y.append(0)
            excel_num += 1
            if excel_num % 10 == 0:
                X = SURF_fake_data_train
                SURF_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                joblib.dump(SURF_clf, savepath + 'SURF_clf.pkl')
                X = []
                Y = []

    # joblib.dump(SURF_clf, savepath + 'SURF_clf.pkl')
    excel_num = 0
    X = []
    Y = []


def test_SURF():
    print('SURF特征SVM分类器测试开始 ……')
    SURF_clf2 = joblib.load(savepath + 'SURF_clf.pkl')
    excel_num = 0
    global accuracy_set
    for test_true_list in test_true_lists:
        if test_true_list.endswith('SURF.xlsx'):
            inputpath = test_true + '/' + test_true_list
            sheets, sheets_1dim = get_SURF(inputpath)
            if excel_num % 10 == 0:
                SURF_true_data_test = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for k in sheets_1dim:
                    SURF_true_data_test.append(k)
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
                for k in sheets_1dim:
                    SURF_fake_data_test.append(k)
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
                for k in sheets_1dim:
                    ELA_true_data_train.append(k)
                    Y.append(1)
            excel_num += 1
            if excel_num % 10 == 0:
                X = ELA_true_data_train
                ELA_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                joblib.dump(ELA_clf, savepath + 'ELA_clf.pkl')
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
                for k in sheets_1dim:
                    ELA_fake_data_train.append(k)
                    Y.append(0)
            excel_num += 1
            if excel_num % 10 == 0:
                X = ELA_fake_data_train
                ELA_clf.partial_fit(X, Y, classes=np.array([0, 1]))
                joblib.dump(ELA_clf, savepath + 'ELA_clf.pkl')
                X = []
                Y = []

    # joblib.dump(ELA_clf, savepath + 'ELA_clf.pkl')
    excel_num = 0
    X = []
    Y = []


def test_ELA():
    print('ELA特征SVM分类器测试开始 ……')
    ELA_clf2 = joblib.load(savepath + 'ELA_clf.pkl')
    excel_num = 0
    global accuracy_set
    for test_true_list in test_true_lists:
        if test_true_list.endswith('ELA.xlsx'):
            inputpath = test_true + '/' + test_true_list
            sheets, sheets_1dim = get_ELA(inputpath)
            if excel_num % 10 == 0:
                ELA_true_data_test = sheets_1dim
                Y = [1]*len(sheets_1dim)
            else:
                for k in sheets_1dim:
                    ELA_true_data_test.append(k)
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
                for k in sheets_1dim:
                    ELA_fake_data_test.append(k)
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
    print('Start Time of {} : {}'.format(string, start))
    function()
    end = datetime.datetime.now()
    print('End Time of {} : {}'.format(string, end))
    print('Running Time of {} : {}\n'.format(string, (end - start)))


if __name__ == '__main__':

    print_runtime(train_color, '训练color特征SVM分类器')
    print_runtime(train_SURF, '训练SURF特征SVM分类器')
    print_runtime(train_ELA, '训练ELA特征SVM分类器')
    print_runtime(test_color, '测试color特征SVM分类器')
    print_runtime(test_SURF, '测试SURF特征SVM分类器')
    print_runtime(test_ELA, '测试ELA特征SVM分类器')

# celeba:3k+1k
# pggan:1.5k+1.5k+1k
# attack:1323+660
# original:1359+675

# ——————————————————————————————————————————————————————————————————————————————————————

# 第一次训练测试，数据是全部数据前一部分训练后一部分测试

# Start Time of 训练color特征SVM分类器 : 2020-05-09 12:13:54.245692
# color特征SVM分类器训练开始 ……
# End Time of 训练color特征SVM分类器 : 2020-05-09 19:58:45.445587
# Running Time of 训练color特征SVM分类器 : 7:44:51.199895

# Start Time of 训练SURF特征SVM分类器 : 2020-05-09 19:58:45.574289
# SURF特征SVM分类器训练开始 ……
# End Time of 训练SURF特征SVM分类器 : 2020-05-09 19:59:33.644728
# Running Time of 训练SURF特征SVM分类器 : 0:00:48.070439

# Start Time of 训练ELA特征SVM分类器 : 2020-05-09 19:59:33.646711
# ELA特征SVM分类器训练开始 ……
# End Time of 训练ELA特征SVM分类器 : 2020-05-09 22:07:43.086718
# Running Time of 训练ELA特征SVM分类器 : 2:08:09.440007

# Start Time of 测试color特征SVM分类器 : 2020-05-10 22:37:42.768791
# color特征SVM分类器测试开始 ……
# color_clf average accuracy: 0.5
# End Time of 测试color特征SVM分类器 : 2020-05-10 01:57:25.119221
# Running Time of 测试color特征SVM分类器 : 3:19:42.350430

# Start Time of 测试SURF特征SVM分类器 : 2020-05-10 01:57:25.285813
# SURF特征SVM分类器测试开始 ……
# SURF_clf average accuracy: 0.5
# End Time of 测试SURF特征SVM分类器 : 2020-05-10 01:57:44.946177
# Running Time of 测试SURF特征SVM分类器 : 0:00:19.660364

# Start Time of 测试ELA特征SVM分类器 : 2020-05-10 01:57:44.947172
# ELA特征SVM分类器测试开始 ……
# ELA_clf average accuracy: 0.5
# End Time of 测试ELA特征SVM分类器 : 2020-05-10 02:42:46.021191
# Running Time of 测试ELA特征SVM分类器 : 0:45:01.074019

# ——————————————————————————————————————————————————————————————————————————————————————

# 第二次训练测试，数据是每一部分数据前一部分训练后一部分测试

# Start Time of 训练color特征SVM分类器 : 2020-05-10 11:53:33.510095
# color特征SVM分类器训练开始 ……
# End Time of 训练color特征SVM分类器 : 2020-05-10 20:27:21.956960
# Running Time of 训练color特征SVM分类器 : 8:33:48.446865

# Start Time of 训练SURF特征SVM分类器 : 2020-05-10 20:27:21.972917
# SURF特征SVM分类器训练开始 ……
# End Time of 训练SURF特征SVM分类器 : 2020-05-10 20:28:25.578129
# Running Time of 训练SURF特征SVM分类器 : 0:01:03.605212

# Start Time of 训练ELA特征SVM分类器 : 2020-05-10 20:28:25.579094
# ELA特征SVM分类器训练开始 ……
# End Time of 训练ELA特征SVM分类器 : 2020-05-10 22:45:25.565611
# Running Time of 训练ELA特征SVM分类器 : 2:16:59.986517

# Start Time of 测试color特征SVM分类器 : 2020-05-10 22:45:25.571563
# color特征SVM分类器测试开始 ……
# color_clf average accuracy: 0.5166666666666667
# End Time of 测试color特征SVM分类器 : 2020-05-11 01:41:00.856418
# Running Time of 测试color特征SVM分类器 : 2:55:35.284855

# Start Time of 测试SURF特征SVM分类器 : 2020-05-11 01:41:00.858413
# SURF特征SVM分类器测试开始 ……
# SURF_clf average accuracy: 0.5166666666666667
# End Time of 测试SURF特征SVM分类器 : 2020-05-11 01:41:18.488249
# Running Time of 测试SURF特征SVM分类器 : 0:00:17.629836

# Start Time of 测试ELA特征SVM分类器 : 2020-05-11 01:41:18.489249
# ELA特征SVM分类器测试开始 ……
# ELA_clf average accuracy: 0.5166666666666667
# End Time of 测试ELA特征SVM分类器 : 2020-05-11 02:26:08.730687
# Running Time of 测试ELA特征SVM分类器 : 0:44:50.241438

# ——————————————————————————————————————————————————————————————————————————————————————

# 准确率三个总是一样，感觉是代码哪里有问题，暂时还没察觉到问题所在，希望有人能看出端倪，欢迎issue！！！
