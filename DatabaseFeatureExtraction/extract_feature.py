import os
import datetime
from CommonFunction.extract_color import draw_color_histogram
from CommonFunction.extract_SURF import draw_SURF
from CommonFunction.extract_ELA import draw_ELA

common_folder_path = 'G:/'

# dataset = 'Celeba_face'
# dataset = 'PGGAN_face'
dataset = 'DFD_face'

input_dataset_path = common_folder_path + dataset
output_dataset_path = common_folder_path + dataset + '_feature'

def extract_feature():
    if not os.path.exists(output_dataset_path):
        os.mkdir(output_dataset_path)
    folder_lists = os.listdir(input_dataset_path)
    # ①
    for folder_list in folder_lists:
        if os.path.isfile(input_dataset_path + '/' + folder_list):
            folder_lists.remove(folder_list)
    for folder_list in folder_lists:
        # There are subfolders in these 3 folders
        if (folder_list != 'img_pggan') and (folder_list != 'attack_c23') and (folder_list != 'original_c23'):
            input_img_folder_path = input_dataset_path + '/' + folder_list
            output_img_folder_path = output_dataset_path + '/' + folder_list
            output_img_color_path = output_img_folder_path + '/color'
            output_img_SURF_path = output_img_folder_path + '/SURF'
            output_img_ELA_path = output_img_folder_path + '/ELA'
            # avoid repeat work
            if not os.path.exists(output_img_folder_path):
                os.mkdir(output_img_folder_path)
                if not os.path.exists(output_img_color_path):
                    os.mkdir(output_img_color_path)
                if not os.path.exists(output_img_SURF_path):
                    os.mkdir(output_img_SURF_path)
                if not os.path.exists(output_img_ELA_path):
                    os.mkdir(output_img_ELA_path)
                img_lists = os.listdir(input_img_folder_path)
                for img_list in img_lists:
                    # ②
                    if not img_list.endswith('.txt'):
                        inputpath = input_img_folder_path + '/' + img_list
                        draw_color_histogram(inputpath , output_img_color_path + '/' + img_list)
                        draw_SURF(inputpath , output_img_SURF_path + '/' + img_list)
                        draw_ELA(inputpath , output_img_ELA_path) 
        else:
            subfolder_lists = os.listdir(input_dataset_path + '/' + folder_list)
            # ③
            for subfolder_list in subfolder_lists:
                if os.path.isfile(input_dataset_path + '/' + folder_list + '/' + subfolder_list):
                    subfolder_lists.remove(subfolder_list)
            for subfolder_list in subfolder_lists:
                input_img_folder_path = input_dataset_path + '/' + folder_list + '/' + subfolder_list
                output_img_folder_path = output_dataset_path + '/' + folder_list
                output_img_subfolder_path = output_dataset_path + '/' + folder_list + '/' + subfolder_list
                output_img_color_path = output_img_subfolder_path + '/color'
                output_img_SURF_path = output_img_subfolder_path + '/SURF'
                output_img_ELA_path = output_img_subfolder_path + '/ELA'
                if not os.path.exists(output_img_folder_path):
                    os.mkdir(output_img_folder_path)
                # avoid repeat work
                if not os.path.exists(output_img_subfolder_path):
                    os.mkdir(output_img_subfolder_path)
                    if not os.path.exists(output_img_color_path):
                        os.mkdir(output_img_color_path)
                    if not os.path.exists(output_img_SURF_path):
                        os.mkdir(output_img_SURF_path)
                    if not os.path.exists(output_img_ELA_path):
                        os.mkdir(output_img_ELA_path)
                    img_lists = os.listdir(input_img_folder_path)
                    for img_list in img_lists:
                        # ④
                        if not img_list.endswith('.txt'):
                            inputpath = input_img_folder_path + '/' + img_list
                            draw_color_histogram(inputpath , output_img_color_path + '/' + img_list)
                            draw_SURF(inputpath , output_img_SURF_path + '/' + img_list)
                            draw_ELA(inputpath , output_img_ELA_path)


if __name__ == '__main__':
    
    startTime = datetime.datetime.now()
    print('startTime: {}'.format(startTime))
    
    extract_feature()

    endTime = datetime.datetime.now()
    print('endTime: {}'.format(endTime))
    print('Running time: {}'.format(endTime - startTime))

# ①~④ avoid processing log.txt file

# Celeba 
# startTime: 2020-02-25 18:08:30.578360
# endTime: 2020-02-25 22:50:47.230785
# Running time: 4:42:16.652425

# PGGAN 
# startTime: 2020-02-25 18:09:01.274854
# endTime: 2020-02-25 22:58:56.398361
# Running time: 4:49:55.123507

# DFD
# startTime: 2020-02-25 18:09:41.216839
# endTime: 2020-02-26 12:08:03.989166
# Running time: 17:58:22.772327