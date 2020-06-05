import numpy as np
import cv2
import os
from multiprocessing import Pool

path = "../../data/connectivity/ori_triangle/"

def filterit(img_name):
    print(img_name)
    img_path = str(path)+str(img_name)
    src = cv2.imread(img_path)
    print(src.shape)
    if src.shape[1]>70 and src.shape[1]<80 and src.shape[0]>80 and src.shape[0]<90:
        os.system('cp %s %s' % (img_path, "../../data/connectivity/real/"+str(img_name)) )
    # a, contours = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(src, contours, -1, (135, 210, 255), 5)  # 参数：图像、轮廓、轮廓序号（负数就画出全部轮廓）、颜色、粗细
    # cv2.imshow("7_contour", src)
    # cv2.moveWindow("7_contour", 300, 10)
    # cv2.waitKey(0)





if __name__ == "__main__":
    list_dirname = os.listdir(path)
    print(type(list_dirname))
    pool = Pool()
    pool.map(filterit, list_dirname)
