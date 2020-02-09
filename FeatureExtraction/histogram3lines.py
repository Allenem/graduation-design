import cv2
import matplotlib.pyplot as plt

image = cv2.imread("./img/meinv.png")
channels = cv2.split(image)
colors = ('b', 'g', 'r')

plt.figure()                # 新建一个图像
plt.title("RGB Histogram")  # 图像的标题
plt.xlabel("Bins")          # X轴标签：亮度0-255
plt.ylabel("# of Pixels")   # Y轴标签：某一亮度的像素数

for (channels, color) in zip(channels, colors):
    hist = cv2.calcHist([channels],  [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()  # 显示图像
