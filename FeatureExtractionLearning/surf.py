
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
img = cv2.imread('img/butterfly.jpg')

surf = cv2.xfeatures2d.SURF_create(8000)
kp = surf.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, None, (0, 255, 0), 4)
 
# 不检查关键点的方向
surf.setUpright(True)
# 修改阈值
surf.setHessianThreshold(30000)
kp = surf.detect(img, None)
img3 = cv2.drawKeypoints(img, kp, None, (0, 255, 0), 4)

plt.subplot(121), plt.imshow(img2[:,:,::-1]),plt.title('Threshold=8000'), plt.axis('off')
plt.subplot(122), plt.imshow(img3[:,:,::-1]),plt.title('Threshold=30000'), plt.axis('off')

plt.tight_layout()
plt.show()