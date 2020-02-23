import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
创建一个SURF对象：
cv2.xfeatures2d.SURF_create(, hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
hessianThreshold：默认100
nOctaves：金字塔组数默认4
nOctaveLayers：每组金子塔的层数默认3
extended：默认False，扩展描述符标志，False表示使用64个元素描述符。
upright：默认False，垂直向上或旋转的特征标志，False表示计算方向。

绘制特征点：
cv2.drawKeypoint(image, keypoints, None, color, flags)
image：输入图像img
keypoints：上面获取的特征点kp
outImage：无输出图像
color：(0, 255, 0) 绿色
flags：绘制点的模式，cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
绘制特征点的时候绘制的是带有方向的圆,这种方法同时显示图像的坐标,size，和方向,是最能显示特征的一种绘制方式。
'''

def draw_SURF(inputpath,outputpath):
    img = cv2.imread(inputpath)
    surf = cv2.xfeatures2d.SURF_create(4000)
    kp = surf.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, (0, 255, 0), 4)

    plt.figure()
    plt.imshow(img2[:,:,::-1])
    plt.title('SURF Threshold=4000')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outputpath) 
    # plt.show()
    plt.close()