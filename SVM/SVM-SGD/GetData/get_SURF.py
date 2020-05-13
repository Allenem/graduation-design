import datetime
import pandas as pd


def get_SURF(inputpath):
    # # print('Excel File Path: {}'.format(inputpath))
    # # start = datetime.datetime.now()
    # return a dict {sheet_name:dataframe}
    excel_file = pd.read_excel(inputpath, sheet_name=None)
    # # end = datetime.datetime.now()
    # # print('Read Excel Time: {}'.format(end - start))
    sheet_num = len(excel_file)
    sheet_names = list(excel_file.keys())  # dict_keys->list
    # print(sheet_names)

    # sheet1 = excel_file[sheet_names[0]]
    # print(type(sheet1))
    # print(len(sheet1))
    # print(sheet1['x'])   # 读取DataFrame某一列
    # print(sheet1.loc[0]) # 读取DataFrame某一行

    sheets = [0]*sheet_num
    sheets_1dim = [0]*sheet_num
    i = 0
    for sheet_name in sheet_names:
        sheets[i] = excel_file[sheet_name].values  # DataFrame->numpy.ndarray
        # sheets[i].flatten() 多维展平为1维。[[…],[…],……,[…]]->[…,…,……,…]
        # 其他方法还有：np.ravel(sheets[i])。[[…],[…],……,[…]]->[…,…,……,…]
        # 或者展平为貌似1维sheets[i].reshape(1,rows*cols)。[[…],[…],……,[…]]->[[…,…,……,…]]
        sheets_1dim[i] = sheets[i].flatten()
        # rows = sheets[i].shape[0]  # 行数
        # cols = sheets[i].shape[1]  # 列数
        # print(rows, cols)
        # print(sheets[i][0][1])
        i += 1
    # print(type(sheets))            # 定义为{}则为dict，每一个key是数字序号，value是矩阵；定义为[0]*sheet_num则为list
    # print(type(sheets_1dim))       # 定义为{}则为dict，每一个key是数字序号，value是一维数组；定义为[0]*sheet_num则为list
    # print(sheets[0])               # 代表第一个sheet，即第一张图片，矩阵
    # print(sheets_1dim[0])          # 代表第一个sheet，即第一张图片，一维数组
    # print(type(sheets[0]))         # <class 'numpy.ndarray'>
    # print(type(sheets_1dim[0]))    # <class 'numpy.ndarray'>

    return sheets, sheets_1dim
