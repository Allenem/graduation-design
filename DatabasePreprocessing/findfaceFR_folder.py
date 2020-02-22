# use face-recognition to detect face from one folder's different folders' images & save them

import os
import time
import datetime
import face_recognition
from PIL import Image

resize_x = 256
resize_y = 256
tempCantFindFaceImgs = []
allCantFindFaceImgsNum = 0
allImgsNum = 0
damagedImg = 0

# Detect face rects
def detect(img, new_path, imglist):
    txt_path = new_path+'/log.txt'
    image = face_recognition.load_image_file(img)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        # print imgs computer can't find face
        '''
        print("I haven't found a face in %s"%(imglist)) 
        '''
        # add can't find face images imglist to tempCantFindFaceImgs
        tempCantFindFaceImgs.append(imglist)
        # add imglist to txt tail
        with open(txt_path, "a", encoding="utf-8") as fi:
            fi.write(imglist+'\n')
        # print txt content line by line once a loop
        '''
        with open(txt_path, encoding="utf-8") as fi:
            lines = fi.readlines()
        print(lines)
        '''
        return []

    # In this case: save the first face found in a pic
    # Get the location of each face in this image
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    resized_face = pil_image.resize((resize_x, resize_y))
    (filename, extension) = os.path.splitext(imglist)
    resized_face.save(new_path+'/FR_'+filename+extension)

if __name__ == '__main__':
    
    start_time = time.clock()
    original_path = 'G:/DFD_img/attack_c23'
    new_path = 'G:/DFD_face/attack_c23'
    log_path = new_path+'/log.txt'
    
    all_file_lists = os.listdir(original_path)
    folder_lists = []

    # find folders in allfilelists
    for all_file_list in all_file_lists:
        (filename, extension) = os.path.splitext(all_file_list)
        if extension == '' :
            folder_lists.append(filename)
    print('# of folders: '+str(len(folder_lists)))
    
    # folder loop
    for folder_list in folder_lists:
        temp_original_path = original_path+'/'+folder_list
        temp_new_path = new_path+'/'+folder_list
        if not os.path.exists(temp_new_path):
            os.mkdir(temp_new_path)
        temp_log_path = new_path+'/'+folder_list+'/log.txt'

        with open(temp_log_path, "a", encoding="utf-8") as fi:
            fi.write('\n# No Found List: \n')

        imglists = os.listdir(temp_original_path) 
        len_imglists = len(imglists)

        for imglist in imglists:
            img = temp_original_path+'/'+imglist
            # exclude error img
            if os.path.getsize(img)  != 0: 
                detect(img, temp_new_path, imglist) 
            else:
                len_imglists -= 1
                damagedImg += 1

        len_tempCantFindFaceImgs = len(tempCantFindFaceImgs)
        error_rate = len_tempCantFindFaceImgs/len_imglists  
        with open(temp_log_path, "a", encoding="utf-8") as fi:
            fi.write('\n# Number of not recognition: {} \n# Number of images: {} \n# Not recognition rate: {}\n'\
            .format(len_tempCantFindFaceImgs,len_imglists,error_rate)) 
        
        # Total
        allCantFindFaceImgsNum += len_tempCantFindFaceImgs
        allImgsNum += len_imglists
        tempCantFindFaceImgs = []

    # Save log.txt
    allErrorRate = allCantFindFaceImgsNum / allImgsNum
    with open(log_path, "a", encoding="utf-8") as fi:
        fi.write('\n# Sum of damaged image: {} \n# Sum of not recognition: {} \
        \n# Sum of not damaged images: {} \n# Total not recognition rate: {}\n'\
        .format(damagedImg,allCantFindFaceImgsNum,allImgsNum,allErrorRate)) 

    end_time = time.clock()
    delta_time = datetime.timedelta(seconds = (end_time-start_time))
    print('Running time using Face-recognition is: %s '%(delta_time))

# OUTPUT1(find face from DFD_img/original_c23)
# # of folders: 363
# Running time using Face-recognition is: 13:46:29.115011

# OUTPUT2(find face from DFD_img/attack_c23)
# # of folders: 3068
# Running time using Face-recognition is: 4 days, 4:05:53.688934