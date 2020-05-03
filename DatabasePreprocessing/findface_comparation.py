import os
import datetime
import cv2
import dlib
import face_recognition

inputpath = 'G:/findface_comparation/original'
outputpath = 'G:/findface_comparation/face'
resize_x = 125
resize_y = 125


def findfaceCV():
    cascade = cv2.CascadeClassifier(
        "E:\Program Files\Python\Python36\Lib\site-packages\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
    imglists = os.listdir(inputpath)
    for imglist in imglists:
        img = cv2.imread(inputpath+'/'+imglist)
        rects = cascade.detectMultiScale(
            img, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            print('No face in {}'.format(imglist))
            for x in range(resize_x):
                for y in range(resize_y):
                    resized_face[x, y] = [0, 0, 0]
            cv2.imwrite(outputpath+'/CV_'+imglist, resized_face)
        else:
            rects[:, 2:] += rects[:, :2]
            for x1, y1, x2, y2 in rects:
                face = img[y1:y2, x1:x2]
                resized_face = cv2.resize(face, (resize_x, resize_y))
                cv2.imwrite(outputpath+'/CV_'+imglist, resized_face)


def findfaceDlib():
    detector = dlib.get_frontal_face_detector()
    imglists = os.listdir(inputpath)
    for imglist in imglists:
        img = cv2.imread(inputpath+'/'+imglist)
        dets = detector(img, 1)
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            face = img[y1:y2, x1:x2]
            resized_face = cv2.resize(face, (resize_x, resize_y))
            cv2.imwrite(outputpath+'/Dlib_'+imglist, resized_face)


def findfaceFR():
    imglists = os.listdir(inputpath)
    for imglist in imglists:
        image = face_recognition.load_image_file(inputpath+'/'+imglist)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            print('No face in {}'.format(imglist))
            for x in range(resize_x):
                for y in range(resize_y):
                    resized_face[x, y] = [0, 0, 0]
            cv2.imwrite(outputpath+'/FR_'+imglist, resized_face)
        else:
            top, right, bottom, left = face_locations[0]
            face = image[top:bottom, left:right]
            resized_face = cv2.resize(face, (resize_x, resize_y))[..., ::-1]
            cv2.imwrite(outputpath+'/FR_'+imglist, resized_face)


if __name__ == '__main__':
    cvbegin = datetime.datetime.now()
    print('OpenCV startTime: {}'.format(cvbegin))
    findfaceCV()
    cvend = datetime.datetime.now()
    print('OpenCV endTime: {}'.format(cvend))
    print('OpenCV running time: {}'.format(cvend - cvbegin))

    dlibbegin = datetime.datetime.now()
    print('Dlib startTime: {}'.format(dlibbegin))
    findfaceDlib()
    dlibend = datetime.datetime.now()
    print('Dlib endTime: {}'.format(dlibend))
    print('Dlib running time: {}'.format(dlibend - dlibbegin))

    FRbegin = datetime.datetime.now()
    print('Face_recognition startTime: {}'.format(FRbegin))
    findfaceFR()
    FRend = datetime.datetime.now()
    print('Face_recognition endTime: {}'.format(FRend))
    print('Face_recognition running time: {}'.format(FRend - FRbegin))
