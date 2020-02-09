"""
hist = cv2.calcHist([image],             # 传入图像（列表）
                    [0],                 # 使用的通道（使用通道：可选[0],[1],[2]）
                    None,                # 没有使用mask(蒙版)
                    [256],               # x范围，柱子的数量
                    [0.0,255.0])         # 像素值范围
                                         # return->list
"""

import cv2    
import numpy as np
import matplotlib.pyplot as plt

def calcAndDrawHist(image, color):  
    hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])  
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)  
    histImg = np.zeros([256,256,3], np.uint8)  
    hpt = int(0.9* 256);  
      
    for h in range(256):  
        intensity = int(hist[h]*hpt/maxVal)  
        cv2.line(histImg,(h,256), (h,256-intensity), color)  
    return histImg


  
original_img = cv2.imread("./img/meinv.png") 
img = cv2.resize(original_img,None,fx=0.6,fy=0.6,interpolation = cv2.INTER_CUBIC)
b, g, r = cv2.split(img)  

histImgB = calcAndDrawHist(b, [255, 0, 0])  
histImgG = calcAndDrawHist(g, [0, 255, 0])  
histImgR = calcAndDrawHist(r, [0, 0, 255])  

'''
# cv2创建图像
cv2.imshow("histImgB", histImgB)  
cv2.imshow("histImgG", histImgG)  
cv2.imshow("histImgR", histImgR)  
cv2.imshow("Img", img)  
cv2.waitKey(0)  
cv2.destroyAllWindows() 
'''


# plt创建图形
plt.figure(1)
plt.gcf().canvas.set_window_title('Test')
plt.gcf().suptitle("Color Histogram")

plt.subplot(2,2,1),plt.imshow(np.flip(original_img,axis = 2)),plt.title('original_img')
plt.subplot(2,2,2),plt.imshow(np.flip(histImgB,axis = 2)),plt.title('histImgB')
plt.subplot(2,2,3),plt.imshow(np.flip(histImgG,axis = 2)),plt.title('histImgG')
plt.subplot(2,2,4),plt.imshow(np.flip(histImgR,axis = 2)),plt.title('histImgR')

plt.tight_layout()  # 调整坐标轴等，使之无重叠
plt.show()