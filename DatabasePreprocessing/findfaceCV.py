# use OpenCV to detect face from images & save them

import os
import time
import datetime
import cv2

resize_x = 256
resize_y = 256
cantFindFaceImgs = []

# Detect face rects
def detect(img, cascade, list):
    txt_path = 'D:/test_face/nofound.txt'
    rects = cascade.detectMultiScale(img, scaleFactor = 1.3, minNeighbors = 4,
                                     flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        # print imgs computer can't find face
        '''
        print("I haven't found a face in %s"%(list)) 
        '''
        # add list to cantFindFaceImgs
        cantFindFaceImgs.append(list)
        # add list to txt tail
        with open(txt_path, "a", encoding="utf-8") as fi:
            fi.write(list+'\n')
        # print txt content line by line once a loop
        '''
        with open(txt_path, encoding="utf-8") as fi:
            lines = fi.readlines()
        print(lines)
        '''
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

if __name__ == '__main__':
    start_time =time.clock()
    # OpenCV Classifier
    cascade = cv2.CascadeClassifier("E:\Program Files\Python\Python36\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
    original_path = 'D:/test'
    new_path = 'D:/test_face'
    # os.listdir show all the filename(including extension)
    imglist = os.listdir(original_path) 


    for list in imglist:
        img = cv2.imread(original_path+'/'+list)
        rects = detect(img, cascade, list)
        if len(rects) != 0:
            for x1, y1, x2, y2 in rects:
                face = img[y1:y2, x1:x2]
                resized_face = cv2.resize(face,(resize_x, resize_y))
                # Save new img, named as original name in new directory, then we can find which are not be detected 
                cv2.imwrite(new_path+'/CV_'+list, resized_face)

    # print("cantFindFaceImgs: %s"%(cantFindFaceImgs))
    print("I have save these images' name that I haven't found a face from in this txt: %s"%(new_path+'/nofound.txt'))
    print("I have save face images in this path: %s"%(new_path))
    end_time = time.clock()
    # get delta time like this 00:00:05.999678
    delta_time = datetime.timedelta(seconds = (end_time-start_time))
    print('Running time using OpenCV is: %s '%(delta_time))