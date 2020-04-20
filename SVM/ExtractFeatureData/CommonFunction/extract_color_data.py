import cv2
import pandas as pd


def extract_color_data(inputpath, outputpath):
    image = cv2.imread(inputpath)
    B, G, R = cv2.split(image)
    # 数组展平
    b = B.ravel()
    g = G.ravel()
    r = R.ravel()
    # 矩阵转置
    channels = list(zip(b, g, r))
    colors = ['b', 'g', 'r']
    dt = pd.DataFrame(channels, columns=colors)
    sheet_name = inputpath.split('/')[-1][:-4]
    dt.to_excel(outputpath, sheet_name=sheet_name, index=0)
