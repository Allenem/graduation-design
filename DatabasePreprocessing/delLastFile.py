import os

def delLastFile():
    original_path = 'D:/test_img'
    all_file_lists = os.listdir(original_path)
    folder_lists = []

    # find folders in allfilelists
    for all_file_list in all_file_lists:
        (filename, extension) = os.path.splitext(all_file_list)
        if extension == '' :
            folder_lists.append(filename)
    print(len(folder_lists))

    # delete last file in each folder
    for folder_list in folder_lists:
        folder_path = original_path+'/'+folder_list
        folder_file_lists =  os.listdir(folder_path)
        folder_file_lists.sort()
        os.remove(folder_path+'/'+folder_file_lists[-1])

if __name__ == '__main_':
    delLastFile()