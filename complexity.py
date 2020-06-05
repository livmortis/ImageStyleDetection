import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import config as cf
from utils.exception import ComplexException

ori_path = "../data/testsym/ori/"
test_show = False

# 1、高斯滤波blur
#    正常图片不需要，会使细节消失，复杂变简单。
#    黑图且非yez（B、G、R三个通道值相同）需要，防止圆形轮廓割裂。
# 2、转灰度图
#    yez图需要

def alpha_bg_to_white(img):
    B_channel = img[:,:,0]
    G_channel = img[:,:,1]
    R_channel = img[:,:,2]
    A_channel = img[:,:,3]
    b_eq_g = np.sum(B_channel) == np.sum(G_channel)
    b_eq_r = np.sum(B_channel) == np.sum(R_channel)

    # 画布改为白色 （画布就是透明的背景部分，也就是alpha等于0的部分，默认这部分的RGB也为0，是“透明的黑色”，改为“透明的白色”）'
    if test_show:
        print(b_eq_r)
        print(b_eq_g)
        print(np.sum(B_channel))
        print(B_channel[0])
        print(np.sum(A_channel))
        print(A_channel[0])
    # if np.sum(B_channel) == 0:
    #     # print("yez")
    #     B_channel[A_channel == 0] = 255
    #     G_channel[A_channel == 0] = 255
    #     R_channel[A_channel == 0] = 255
    #     # ret, img = cv2.threshold(img, thresh=254, maxval=255, type=cv2.THRESH_BINARY_INV)
    #
    #     # b, g, r, a = cv2.split(img)
    #     # img = cv2.merge([r, g, b, a])
    #     # img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    if (np.sum([B_channel, G_channel, R_channel]) == 0)  or \
             (np.sum(B_channel) == np.sum(G_channel) ):
        # print('it is')
        b_converted = 255-img[:, :, 3]+img[:,:,0]
        g_converted = 255-img[:, :, 3]+img[:,:,1]
        r_converted = 255-img[:, :, 3]+img[:,:,2]

        img[:, :, 0] = b_converted
        img[:, :, 0][b_converted > 255] = 255

        img[:, :, 1] = g_converted
        img[:, :, 1][g_converted > 255] = 255

        img[:, :, 2] = r_converted
        img[:, :, 2][r_converted > 255] = 255


    if b_eq_g and b_eq_r:
        # print("fuz")
        img = cv2.blur(img, (1, 1), 10)         # 高斯滤波，解决了一些完整轮廓被检测的支离破碎的问题。
                        #适合纯黑的已被割裂的圆形轮廓， 不适合一些细节比较近似的图。
                        #yez需要在前   10706.png

        if test_show:
            cv2.imshow("img after blur : ", img)
            cv2.waitKey(0)
    return img


def compJudge(listdir, mode, gy, gyid):
    try:
        if gy != -1:
            if gy == cf.CF or gy ==cf.XTYS or gy == cf.ZW or gy == cf.DCX  :
                return '复杂'

        for img_name in listdir:
            if mode:
                img_path = str(ori_path) + str(img_name)
            else:
                img_path = img_name
            src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 1 3 6 10
            imgshape = src.shape
            # src = cv2.resize(src, (512, int(512 * (imgshape[0]/imgshape[1])) ))
            # print("shape is : "+str(src.shape))
            H = src.shape[0]
            W = src.shape[1]
            if test_show:
                cv2.imshow("src : ", src)
                cv2.waitKey(0)
            img = alpha_bg_to_white(src)

            # 黑色复杂元素 + 普通元素 ： 不需要转gray
            # yez需要
            # b, g, r, a = cv2.split(img)
            # img = cv2.merge([r, g, b, a])

            # img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            img = cv2.Canny(img, 100, 400)

            if test_show:
                cv2.imshow("img after canny : ", img)
                cv2.waitKey(0)
            i, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE 所有边界点, RETR_LIST不建立等级关系
            contours = np.array(contours)
            cv2.drawContours(img, contours, -1, (255, 180, 255), 2)  # 参数：图像、轮廓、轮廓序号（负数就画出全部轮廓）、颜色、粗细
            if test_show:
                cv2.imshow("contour : ", img)
                cv2.waitKey(0)

            if not contours.shape[0] == 1:
                contours = np.squeeze(contours)
            else :
                contours = np.squeeze(contours)
                contours = np.array([contours])
            count_num = contours.shape[0]
            # print("count_num: "+str(count_num))
            if count_num <= 7:
                if mode:
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/conplexity/simple'))
                else:
                    return '简单'
            elif count_num <30:
                if mode:
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/conplexity/common'))
                else:
                    return '复杂度一般'
            else:
                if mode:
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/conplexity/complex'))
                else:
                    return '复杂'
    except Exception as e :
        raise ComplexException(str(e))


if __name__ == "__main__":
    listdir = os.listdir(ori_path)
    mode = 0     # 0 演示单图  1 多图分批
    if test_show:
        listdir = ['10260.png']
    compJudge(listdir, mode, -1)
