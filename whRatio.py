import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


ori_path = "../data/testsym/ori/"       # 用检测对称的原图

def classifyRatio(listdir, mode, gy, gyid):

    for img_name in listdir:
        # print(img_name)
        if mode:
            img_path = str(ori_path) + str(img_name)
        else:
            img_path = img_name


        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        H = img.shape[0]
        W = img.shape[1]
        if H >= W * 1.3:
            # print("瘦长")
            if mode:
                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/whRatio/h'))
            else:
                return "瘦"
        elif W >= H * 1.25:
            # print("扁")
            if mode:
                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/whRatio/w'))
            else:
                return "扁"
        else:
            # print('normal')
            if mode:
                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/whRatio/normal'))
            else:
                return "长宽适中"

if __name__ == "__main__":
    listdir = os.listdir(ori_path)
    mode = 0     # 0 演示单图  1 多图分批
    # listdir = ["10997.png"]
    classifyRatio(listdir, mode)






















