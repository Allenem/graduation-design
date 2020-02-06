# 针对Deepfake假脸视频面部细节特征的提取算法

## 目录

- [仓库说明](#一仓库说明)
- [工作计划](#二工作计划)
- [调研和资料分析](#三调研和资料分析)
- [学习特征提取](#四学习特征提取)
- [数据库预处理](#五数据库预处理)
- [提取特征并检测GAN真假脸差异](#六提取特征并检测GAN真假脸差异)
- [Deepfake换脸检测算法实现](#七Deepfake换脸检测算法实现)
- [完成论文](#八完成论文)

## 一、仓库说明

```bash
.
│  LICENSE                # 许可说明
│  README.md              # 说明文件
│
├─Preparation             # 阅读相关论文，Python学习，环境准备工作
├─FeatureExtraction       # 学习特征提取 代码文件夹
├─DatabasePreprocessing   # 数据库预处理，提取人脸  代码文件夹
├─DetectGANDifferences    # 提取特征并检测GAN真假脸差异  代码文件夹
├─DeepfakeDetection       # 算法实现 Deepfake 换脸检测  代码文件夹
└─Paper                   # 我的论文
```

## 二、工作计划

### 1.数据库分配

学生 | 真脸 | GAN假脸数据库 | Deepfake数据库
-|-|-|-
Allenem | Celeba(validation,test) | PGGA | DeepfakeDetection

### 2.特征分配

学生 | 特征
-|-
Allenem | 1、颜色直方图 2、Surf 3、错误级别分析（Error level analysis，ELA）

### 3.工作计划

起止时间 | 工作内容 | 备注
-|-|-
2020.01-2020.02 | 调研和资料分析 | 	
2020.01-2020.02	| 数据库预处理 | 视频分帧、人脸提取及定位
2020.02-2020.03 | 提取人脸特征、检测GAN真假脸图像差异 | 隐写分析特征或者图像篡改特征
2020.03-2020.04 | Deepfake换脸视频检测算法实现 | SVM分类器等不同分类器
2020.04-2020.05 | 完成毕业论文 |

## 三、调研和资料分析

### 1.参考文献

[[1] Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)

[[2] Deepfake Video Detection through Optical Flow Based CNN](openaccess.thecvf.com/content_ICCVW_2019/papers/HBU/Amerini_Deepfake_Video_Detection_through_Optical_Flow_Based_CNN_ICCVW_2019_paper.pdf)

[[3] Exposing deep fakes using inconsistent head poses](https://arxiv.org/pdf/1811.00661.pdf)

[[4] Exposing GAN-synthesized Faces Using Landmark Locations](https://arxiv.org/pdf/1904.00167.pdf)

[[5] Perceptual Judgments to Detect ComputerGenerated Forged Faces in Social Media](https://kopernio.com/viewer?doi=10.1007/978-3-030-20984-1_4&token=WzE3MDUwMzYsIjEwLjEwMDcvOTc4LTMtMDMwLTIwOTg0LTFfNCJd.EPCnRwtIa113H6qoV-aTHHQoOOs)

### 2.我自己的中文翻译

参考文件夹 `Preparation`

### 3.Python学习和人脸检测学习

学习笔记： https://github.com/Allenem/GitHubNoteBook#python

OpenCV，dlib，face_recognition 实现人脸检测，标志检测等实验小测试： https://github.com/Allenem/PyTest

## 四、学习特征提取



## 五、数据库预处理



## 六、提取特征并检测GAN真假脸差异



## 七、Deepfake换脸检测算法实现



## 八、完成论文


