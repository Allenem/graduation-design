import os
import datetime
from CommonFunction.extract_color import draw_color_histogram
from CommonFunction.extract_SURF import draw_SURF
from CommonFunction.extract_ELA import draw_ELA

common_folder_path = 'G:/Feature/'

# dataset = 'test'
# dataset = 'Celeba'
# dataset = 'PGGAN'
dataset = 'DFD'

input_dataset_path = common_folder_path + dataset
output_dataset_path = common_folder_path + dataset + '_feature'

def extract_feature():
    if not os.path.exists(output_dataset_path):
        os.mkdir(output_dataset_path)
    folder_lists = os.listdir(input_dataset_path)
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
                    inputpath = input_img_folder_path + '/' + img_list
                    draw_color_histogram(inputpath , output_img_color_path + '/' + img_list)
                    draw_SURF(inputpath , output_img_SURF_path + '/' + img_list)
                    draw_ELA(inputpath , output_img_ELA_path) 
        else:
            subfolder_lists = os.listdir(input_dataset_path + '/' + folder_list)
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

# # test
# startTime: 2020-02-24 14:13:47.731645
# endTime: 2020-02-24 14:13:54.609752
# Running time: 0:00:06.878107

# # Celeba 
# startTime: 2020-02-24 14:16:13.824223
# endTime: 2020-02-24 14:16:46.069675
# Running time: 0:00:32.245452

# # PGGAN 
# startTime: 2020-02-24 14:20:15.186285
# endTime: 2020-02-24 14:21:47.677183
# Running time: 0:01:32.490898

# # DFD
# startTime: 2020-02-24 14:22:45.027296
# endTime: 2020-02-24 14:59:50.830777
# Running time: 0:37:05.803481