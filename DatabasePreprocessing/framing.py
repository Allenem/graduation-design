import os
import cv2
import time
import datetime

def framing():
    input_path = 'G:/DFD/attack_c23/videos'
    output_path = 'G:/DFD_img/attack_c23'
    txt_path = output_path+'/log.txt'
    with open(txt_path, "a", encoding="utf-8") as fi:
        fi.write('\n AllVideosFullName \t Index \t Frame \t Picture\n')

    videos = os.listdir(input_path)
    videos.sort(key = lambda x: x[:-4])

    if len(videos) != 0:
        video_num = 0
        for each_video in videos:
            # print('Video {} is running ...'.format(video_num))
            each_video_input = input_path+'/'+str(each_video)
            each_video_output_path = output_path+'/'+str(each_video[:-4])
            if not os.path.exists(each_video_output_path):
                os.mkdir(each_video_output_path)
            
            capture = cv2.VideoCapture(each_video_input)
            if capture.isOpened():
                real = True
            else:
                real = False

            frame_step = 10
            frame_num = 0
            picture_num = 0

            while real:
                real, frame = capture.read()
                if(frame_num % frame_step == 0):
                    cv2.imwrite(each_video_output_path+'/Frame'+str(frame_num)+'_Pic'+str(picture_num)+'.jpg',frame)
                    picture_num += 1
                frame_num += 1
                # cv2.waitKey(1)

            video_num += 1
            with open(txt_path, "a", encoding="utf-8") as fi:
                fi.write('{} \t {} \t {} \t {}\n'.format(each_video[:-4], video_num, frame_num, picture_num ))
            capture.release()

        print('Running log has been saved here: '+txt_path)

    else:
        print('Empty Directory')


if __name__ == '__main__':
    start_time = time.clock()
    framing()
    end_time = time.clock()
    delta_time = datetime.timedelta(seconds = (end_time-start_time))
    print('Running time is: %s '%(delta_time))


# Running log has been saved here: G:/DFD_img/original_c23/log.txt
# Running time is: 1:05:49.907241

