import os
import datetime
import pandas as pd
from CommonFunction.extract_color_data import extract_color_data
from CommonFunction.extract_SURF_data import extract_SURF_data
from CommonFunction.extract_ELA_data import extract_ELA_data

# init some variables
pathlist = ['Celeba', 'PGGAN', 'DFD']
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


def print_time(inputfiles, inputpath_new, outputfile_color, outputfile_SURF, outputfile_ELA, outputpath_ELA):
    color_startTime = datetime.datetime.now()
    print('Color startTime: {}'.format(color_startTime))
    color(inputfiles, inputpath_new, outputfile_color)
    color_endTime = datetime.datetime.now()
    print('Color endTime: {}'.format(color_endTime))
    print('Color running time: {}'.format(color_endTime - color_startTime))

    SURF_startTime = datetime.datetime.now()
    print('SURF startTime: {}'.format(SURF_startTime))
    SURF(inputfiles, inputpath_new, outputfile_SURF)
    SURF_endTime = datetime.datetime.now()
    print('SURF endTime: {}'.format(SURF_endTime))
    print('SURF running time: {}'.format(SURF_endTime - SURF_startTime))

    ELA_startTime = datetime.datetime.now()
    print('ELA startTime: {}'.format(ELA_startTime))
    ELA(inputfiles, inputpath_new, outputfile_ELA, outputpath_ELA)
    ELA_endTime = datetime.datetime.now()
    print('ELA endTime: {}'.format(ELA_endTime))
    print('ELA running time: {}'.format(ELA_endTime - ELA_startTime))


def main():
    for inputpath in inputpaths:
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
                print('inputsubfolders:')
                print(inputsubfolders)

                # inputsubfolders[0](eg. PGGAN_face/img_pggan/zip000000, ... folders) is not endswith '.jpg'
                if not inputsubfolders[0].endswith('.jpg'):
                    if not os.path.exists(outputpath + inputfolder):
                        os.mkdir(outputpath + inputfolder)
                    for inputsubfolder in inputsubfolders:
                        inputfiles = os.listdir(
                            inputpath + inputfolder + '/' + inputsubfolder)
                        print('inputfiles:')
                        print(inputfiles)
                        # processing ...
                        inputpath_new = inputpath + inputfolder + '/' + inputsubfolder + '/'
                        outputfile_color = outputpath + inputfolder + \
                            '/' + inputsubfolder + '_color.xlsx'
                        outputfile_SURF = outputpath + inputfolder + '/' + inputsubfolder + '_SURF.xlsx'
                        outputfile_ELA = outputpath + inputfolder + '/' + inputsubfolder + '_ELA.xlsx'
                        outputpath_ELA = outputpath + inputfolder + '/'
                        print_time(inputfiles, inputpath_new, outputfile_color,
                                   outputfile_SURF, outputfile_ELA, outputpath_ELA)

                # inputsubfolders[0](eg. Celeba/train/000000.jpg, 000001.jpg, ... files) is endswith '.jpg'
                elif inputsubfolders[0].endswith('.jpg'):
                    inputfiles = inputsubfolders
                    print('inputfiles:')
                    print(inputfiles)
                    # processing ...
                    inputpath_new = inputpath + inputfolder + '/'
                    outputfile_color = outputpath + inputfolder + '_color.xlsx'
                    outputfile_SURF = outputpath + inputfolder + '_SURF.xlsx'
                    outputfile_ELA = outputpath + inputfolder + '_ELA.xlsx'
                    outputpath_ELA = outputpath + '/'
                    print_time(inputfiles, inputpath_new, outputfile_color,
                               outputfile_SURF, outputfile_ELA, outputpath_ELA)

            elif inputfolder.endswith('.jpg'):
                inputfiles = inputfolders
                print('inputfiles:')
                print(inputfiles)
                # processing ... BUT this case is NOT exist


if __name__ == '__main__':
    main()
