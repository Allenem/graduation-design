# SVM 分类真假脸

## 文件目录介绍

```bash
SVM:.
│  README.md                    # 该文件夹说明文件
│  svm_ref_streamvalfile.py     # jyt同学的参考代码
│
├─ExtractFeatureData            # 特征数据提取代码文件夹
│  │  extract_feature_data.py   # 特征数据提取主程序
│  │  OUTPUT.txt                # 部分运行日志
│  │  test.py                   # 特征数据提取主程序之前的测试代码
│  │
│  └─CommonFunction             # 公用函数，分别提取特征并存入excel的一个sheet
│        extract_color_data.py
│        extract_SURF_data.py
│        extract_ELA_data.py
│
├─SVM-SGD                       # SGD(Stochastic Gradient Descent)
│  │  svm_SGD.py                # 随机梯度下降分类器主程序(含训练、测试代码)
│  │
│  └─GetData                    # 从excel中提取数据返回一维列表，3者基本一样
│        get_color.py
│        get_SURF.py            # 3者中最先写的，注释详细
│        get_ELA.py
│
└─SVM-test                      # SVM学习文件夹
        model.pickle            # 保存的svm_learning.py训练模型
        svm_eg.py               # 基本的SVM学习代码
        svm_learning.py         # 一个SVM分类例子
```