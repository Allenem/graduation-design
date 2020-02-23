import os
from CommonFunction.extract_color import draw_color_histogram
from CommonFunction.extract_SURF import draw_SURF
from CommonFunction.extract_ELA import draw_ELA

common_folder_path = 'G:/Feature/'

dataset = 'test'
# dataset = 'Celeba'
# dataset = 'PGGAN'
# dataset = 'DFD'

input_dataset_path = common_folder_path + dataset
output_dataset_path = common_folder_path + dataset + '_feature'
folder_lists = os.listdir(input_dataset_path)

def extract_feature():
    for folder_list in folder_lists:
        input_img_folder_path = input_dataset_path +'/' + folder_list
        output_img_folder_path = output_dataset_path +'/' + folder_list
        if not os.path.exists(output_img_folder_path):
            os.mkdir(output_img_folder_path)

        output_img_color_path = output_img_folder_path +'/color'
        if not os.path.exists(output_img_color_path):
            os.mkdir(output_img_color_path)
        output_img_SURF_path = output_img_folder_path +'/SURF'
        if not os.path.exists(output_img_SURF_path):
            os.mkdir(output_img_SURF_path)
        output_img_ELA_path = output_img_folder_path +'/ELA'
        if not os.path.exists(output_img_ELA_path):
            os.mkdir(output_img_ELA_path)
            
        img_lists = os.listdir(input_img_folder_path)
        for img_list in img_lists:
            inputpath = input_img_folder_path + '/' + img_list
            draw_color_histogram(inputpath , output_img_color_path + '/' + img_list)
            draw_SURF(inputpath , output_img_SURF_path + '/' + img_list)
            draw_ELA(inputpath , output_img_ELA_path)


if __name__ == '__main__':
    extract_feature()