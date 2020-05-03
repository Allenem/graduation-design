import datetime
import pandas as pd


def get_ELA(inputpath):
    print('Excel File Path: {}'.format(inputpath))
    start = datetime.datetime.now()
    # return a dict {sheet_name:dataframe}
    excel_file = pd.read_excel(inputpath, sheet_name=None)
    end = datetime.datetime.now()
    print('Read Excel Time: {}'.format(end - start))
    sheet_num = len(excel_file)
    sheet_names = list(excel_file.keys())  # dict_keys->list

    sheets = [0]*sheet_num
    sheets_1dim = [0]*sheet_num
    i = 0
    for sheet_name in sheet_names:
        sheets[i] = excel_file[sheet_name].values  # DataFrame->numpy.ndarray
        sheets_1dim[i] = sheets[i].flatten()
        i += 1

    return sheets, sheets_1dim
