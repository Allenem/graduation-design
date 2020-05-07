import os
import datetime
import pandas as pd
from CommonFunction.extract_color_data import extract_color_data
from CommonFunction.extract_SURF_data import extract_SURF_data
from CommonFunction.extract_ELA_data import extract_ELA_data

# init some variables
# pathlist = ['Celeba', 'PGGAN', 'DFD']
pathlist = ['DFD']
inputpaths = ['G:/SVM/' + i + '_face/' for i in pathlist]


def color(inputfiles, inputpath_new, outputfile_color):
    excel_writer = pd.ExcelWriter(outputfile_color)
    for inputfile in inputfiles:
        inputpath_file = inputpath_new + inputfile
        # print(inputpath_file)
        extract_color_data(inputpath_file, excel_writer)
    excel_writer.save()
    excel_writer.close()


def SURF(inputfiles, inputpath_new, outputfile_SURF):
    excel_writer = pd.ExcelWriter(outputfile_SURF)
    for inputfile in inputfiles:
        inputpath_file = inputpath_new + inputfile
        # print(inputpath_file)
        extract_SURF_data(inputpath_file, excel_writer)
    excel_writer.save()
    excel_writer.close()


def ELA(inputfiles, inputpath_new, outputfile_ELA, outputpath_ELA):
    excel_writer = pd.ExcelWriter(outputfile_ELA)
    for inputfile in inputfiles:
        inputpath_file = inputpath_new + inputfile
        # print(inputpath_file)
        extract_ELA_data(inputpath_file, outputpath_ELA, excel_writer)
    excel_writer.save()
    excel_writer.close()


def process_per10(inputfiles, inputpath_new, outputfile_color, outputfile_SURF, outputfile_ELA, outputpath_ELA):
    # color_startTime = datetime.datetime.now()
    # print('Color startTime: {}'.format(color_startTime))
    color(inputfiles, inputpath_new, outputfile_color)
    # color_endTime = datetime.datetime.now()
    # print('Color endTime: {}'.format(color_endTime))
    # print('Color running time: {}'.format(color_endTime - color_startTime))

    # SURF_startTime = datetime.datetime.now()
    # print('SURF startTime: {}'.format(SURF_startTime))
    SURF(inputfiles, inputpath_new, outputfile_SURF)
    # SURF_endTime = datetime.datetime.now()
    # print('SURF endTime: {}'.format(SURF_endTime))
    # print('SURF running time: {}'.format(SURF_endTime - SURF_startTime))

    # ELA_startTime = datetime.datetime.now()
    # print('ELA startTime: {}'.format(ELA_startTime))
    ELA(inputfiles, inputpath_new, outputfile_ELA, outputpath_ELA)
    # ELA_endTime = datetime.datetime.now()
    # print('ELA endTime: {}'.format(ELA_endTime))
    # print('ELA running time: {}'.format(ELA_endTime - ELA_startTime))


def main():
    for inputpath in inputpaths:
        startTime = datetime.datetime.now()
        print('{} startTime: {}'.format(inputpath, startTime))

        outputpath = inputpath[:-6] + '_feature_data/'
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)
        inputfolders = os.listdir(inputpath)
        print('inputfolders:')
        print(inputfolders)
        inputsubfolders = []
        inputfiles = []

        # main program
        for inputfolder in inputfolders:
            # inputfolder(eg. Celeba/train, test, ... folders) is not endswith '.jpg'
            if not inputfolder.endswith('.jpg'):
                inputsubfolders = os.listdir(inputpath + inputfolder)

                # inputsubfolders[0](eg. PGGAN_face/img_pggan/zip000000, ... folders) is not endswith '.jpg'
                if not inputsubfolders[0].endswith('.jpg'):
                    print('inputsubfolders:')
                    print(inputsubfolders)
                    if not os.path.exists(outputpath + inputfolder):
                        os.mkdir(outputpath + inputfolder)
                    for inputsubfolder in inputsubfolders:
                        # define some inputs
                        inputfiles = os.listdir(
                            inputpath + inputfolder + '/' + inputsubfolder)
                        # print('inputfiles:')
                        # print(inputfiles)
                        inputfiles_new = [inputfiles[i:i + 10]
                                          for i in range(0, len(inputfiles), 10)]
                        for (index, item) in enumerate(inputfiles_new):
                            print('index:{}'.format(index))
                            inputpath_new = inputpath + inputfolder + '/' + inputsubfolder + '/'
                            outputfile_color = outputpath + inputfolder + \
                                '/' + inputsubfolder + '_' + \
                                str(index) + '_color.xlsx'
                            outputfile_SURF = outputpath + inputfolder + '/' + \
                                inputsubfolder + '_' + \
                                str(index) + '_SURF.xlsx'
                            outputfile_ELA = outputpath + inputfolder + '/' + \
                                inputsubfolder + '_' + str(index) + '_ELA.xlsx'
                            outputpath_ELA = outputpath + inputfolder + '/'
                            # processing ...
                            process_per10(item, inputpath_new, outputfile_color,
                                          outputfile_SURF, outputfile_ELA, outputpath_ELA)

                # inputsubfolders[0](eg. Celeba/train/000000.jpg, 000001.jpg, ... files) is endswith '.jpg'
                elif inputsubfolders[0].endswith('.jpg'):
                    # define some inputs
                    inputfiles = inputsubfolders
                    # print('inputfiles:')
                    # print(inputfiles)  # eg. [x1,…,x10,x11,…,x20,x21,…,x25]
                    inputfiles_new = [inputfiles[i:i + 10]
                                      for i in range(0, len(inputfiles), 10)]
                    # print('inputfiles_new:')
                    # print(inputfiles_new)  # eg. [[x1,…,x10],[x11,…,x20],[x21,…,x25]]
                    for (index, item) in enumerate(inputfiles_new):
                        # print(index, item)  # eg. 0 [x1,…,x10]
                        print('index:{}'.format(index))
                        inputpath_new = inputpath + inputfolder + '/'
                        outputfile_color = outputpath + \
                            inputfolder + '_' + str(index) + '_color.xlsx'
                        outputfile_SURF = outputpath + \
                            inputfolder + '_' + str(index) + '_SURF.xlsx'
                        outputfile_ELA = outputpath + \
                            inputfolder + '_' + str(index) + '_ELA.xlsx'
                        outputpath_ELA = outputpath + '/'
                        # processing ...
                        process_per10(item, inputpath_new, outputfile_color,
                                      outputfile_SURF, outputfile_ELA, outputpath_ELA)

            elif inputfolder.endswith('.jpg'):
                inputfiles = inputfolders
                print('inputfiles:')
                print(inputfiles)
                # processing ... BUT this case is NOT exist

        endTime = datetime.datetime.now()
        print('{} endTime: {}'.format(inputpath, endTime))
        print('{} running time: {}'.format(inputpath, endTime - startTime))


if __name__ == '__main__':
    main()


# OUTPUT:

# G:/SVM/Celeba_face/ startTime: 2020-05-07 00:43:19.002506
# inputfolders:
# ['devel', 'test']
# index: 0
# ……
# index:299
# index: 0
# ……
# index:99
# G:/SVM/Celeba_face/ endTime: 2020-05-07 09:40:59.194924
# G:/SVM/Celeba_face/ running time: 8:57:40.192418

# G:/SVM/PGGAN_face/ startTime: 2020-05-07 09:40:59.228836
# inputfolders:
# ['devel', 'test', 'train']
# index: 0
# ……
# index:149
# index:0
# ……
# index:99
# index:0
# ……
# index:149
# G:/SVM/PGGAN_face/ endTime: 2020-05-07 18:39:00.194182
# G:/SVM/PGGAN_face/ running time: 8:58:00.965346

# G:/SVM/DFD_face/ startTime: 2020-05-07 18:52:14.036170
# inputfolders:
# ['attack_c23', 'original_c23']
# inputsubfolders:
# ……
# inputsubfolders:
# ……
