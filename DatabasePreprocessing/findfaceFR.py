# use face-recognition to detect face from images & save them

import os
import time
import datetime
import face_recognition
from PIL import Image

resize_x = 256
resize_y = 256
cantFindFaceImgs = []

# Detect face rects
def detect(img, new_path, list):
    txt_path = new_path+'/nofound.txt'
    image = face_recognition.load_image_file(img)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        # print imgs computer can't find face
        '''
        print("I haven't found a face in %s"%(list)) 
        '''
        # add can't find face images list to cantFindFaceImgs
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
    
    # In this case: save all faces found in a pic
    '''
    for i,face_location in enumerate(face_locations):

        # Get the location of each face in this image
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        resized_face = pil_image.resize((resize_x, resize_y))
        (filename, extension) = os.path.splitext(list)
        resized_face.save(new_path+'/FR_'+filename+'_'+str(i)+extension)
    '''

    # In this case: save the first face found in a pic
    # Get the location of each face in this image
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    resized_face = pil_image.resize((resize_x, resize_y))
    (filename, extension) = os.path.splitext(list)
    resized_face.save(new_path+'/FR_'+filename+extension)

if __name__ == '__main__':
    start_time =time.clock()
    original_path = 'D:/Celeba/test'
    new_path = 'D:/Celeba_face/test'
    # os.listdir show all the filename(including extension)
    imglist = os.listdir(original_path) 
    len_imglist = len(imglist)

    for list in imglist:
        img = original_path+'/'+list
        detect(img, new_path, list)

    # print can't find face images list: cantFindFaceImgs
    # print("cantFindFaceImgs: %s"%(cantFindFaceImgs))
    len_cantFindFaceImgs = len(cantFindFaceImgs)
    print("I have save these images' name that I haven't found a face from in this txt: %s"%(new_path+'/nofound.txt'))
    print("I have save face images in this path: %s"%(new_path))
    print("Face-recognition accuracy: %f %"%(len_cantFindFaceImgs/len_imglist))
    end_time = time.clock()
    # get delta time like this 00:00:05.999678
    delta_time = datetime.timedelta(seconds = (end_time-start_time))
    print('Running time using Face-recognition is: %s '%(delta_time))