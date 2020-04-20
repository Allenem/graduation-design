import os
import datetime
import pandas as pd
from CommonFunction.extract_color_data import extract_color_data
from CommonFunction.extract_SURF_data import extract_SURF_data
from CommonFunction.extract_ELA_data import extract_ELA_data

inputpath = 'G:/SVM/Celeba_face/devel/'
inputfiles = os.listdir(inputpath)
outputpath = 'G:/SVM/Celeba_feature_data/'


def color():
    outputfile = outputpath + 'devel_color.xlsx'
    excel_writer = pd.ExcelWriter(outputfile)
    for inputfile in inputfiles:
        inputpath_file = inputpath + inputfile
        print(inputpath_file)
        extract_color_data(inputpath_file, excel_writer)
    excel_writer.save()
    excel_writer.close()


def SURF():
    outputfile = outputpath + 'devel_SURF.xlsx'
    excel_writer = pd.ExcelWriter(outputfile)
    for inputfile in inputfiles:
        inputpath_file = inputpath + inputfile
        print(inputpath_file)
        extract_SURF_data(inputpath_file, excel_writer)
    excel_writer.save()
    excel_writer.close()


def ELA():
    outputfile = outputpath + 'devel_ELA.xlsx'
    excel_writer = pd.ExcelWriter(outputfile)
    for inputfile in inputfiles:
        inputpath_file = inputpath + inputfile
        print(inputpath_file)
        extract_ELA_data(inputpath_file, outputpath, excel_writer)
    excel_writer.save()
    excel_writer.close()


if __name__ == '__main__':
    # color_startTime = datetime.datetime.now()
    # print('Color startTime: {}'.format(color_startTime))
    # color()
    # color_endTime = datetime.datetime.now()
    # print('Color endTime: {}'.format(color_endTime))
    # print('Color running time: {}'.format(color_endTime - color_startTime))

    # SURF_startTime = datetime.datetime.now()
    # print('SURF startTime: {}'.format(SURF_startTime))
    # SURF()
    # SURF_endTime = datetime.datetime.now()
    # print('SURF endTime: {}'.format(SURF_endTime))
    # print('SURF running time: {}'.format(SURF_endTime - SURF_startTime))

    ELA_startTime = datetime.datetime.now()
    print('ELA startTime: {}'.format(ELA_startTime))
    ELA()
    ELA_endTime = datetime.datetime.now()
    print('ELA endTime: {}'.format(ELA_endTime))
    print('ELA running time: {}'.format(ELA_endTime - ELA_startTime))
