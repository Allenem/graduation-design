import cv2
import pandas as pd


def extract_SURF_data(inputpath, outputpath):
    img = cv2.imread(inputpath)
    surf = cv2.xfeatures2d.SURF_create(4000)
    kps, features = surf.detectAndCompute(img, None)
    kps_data = []
    for kp in kps:
        # 关键点X，Y，从左到右0~255，从上到下0~255。关键点角度。关键点直径大小
        # print(kp.pt[0], kp.pt[1], kp.angle, kp.size)
        kps_data.append([kp.pt[0], kp.pt[1], kp.angle, kp.size])

    # 统一特征点为15个
    kps_data_len = len(kps_data)
    if kps_data_len < 15:
        for i in range(15-kps_data_len):
            kps_data.append([0, 0, 0, 0])
    elif kps_data_len > 15:
        del kps_data[15:]
    # 按照半径排序
    kps_data.sort(key=lambda x: x[3], reverse=True)
    titles = ['x', 'y', 'angle', 'diameter']
    dt = pd.DataFrame(kps_data, columns=titles)
    sheet_name = inputpath.split('/')[-1][:-4]
    dt.to_excel(outputpath, sheet_name=sheet_name, index=0)
