# 三角形

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

ori_path = "../data/connectivity/ori_triangle/"



def alpha_bg_to_white(img):
    B_channel = img[:,:,0]
    G_channel = img[:,:,1]
    R_channel = img[:,:,2]
    A_channel = img[:,:,3]
    # 画布改为白色 （画布就是透明的背景部分，也就是alpha等于0的部分，默认这部分的RGB也为0，是“透明的黑色”，改为“透明的白色”）
    B_channel[ A_channel == 0 ] = 255
    G_channel[ A_channel == 0 ] = 255
    R_channel[ A_channel == 0 ] = 255

    # ret, img = cv2.threshold(img, thresh=254, maxval=255, type=cv2.THRESH_BINARY_INV)
    return img

def save_formal_base_npy(src):
    img = alpha_bg_to_white(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    edge_dots = []
    i, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE 所有边界点, RETR_LIST不建立等级关系
    # (185, 1, 2)
    for i, con in enumerate(contours):
        if con.shape[0] == 185:
            dots = np.squeeze(con)
            cv2.drawContours(img, con, -1, (135, 210, 255), 5)  # 参数：图像、轮廓、轮廓序号（负数就画出全部轮廓）、颜色、粗细
            # print(dots.shape)
            for dot in dots:
                if i==2:
                    if dot[0] != 86 and dot[1] != 0:
                        x = dot[0]
                        y = dot[1]
                        edge_dots.append([y,x-2])    # src在该点应该为0
                elif i==3:
                    if dot[0] != 0 and dot[1] != 0:
                        x = dot[0]
                        y = dot[1]
                        edge_dots.append([y, x+2])  # src在该点应该为0
    edge_dots_np = np.array(edge_dots)
    np.save(formal_triangle_npy, edge_dots_np)  # 长度140


def save_left_base_npy(src):
    img = alpha_bg_to_white(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    edge_dots = []
    i, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE 所有边界点, RETR_LIST不建立等级关系
    # (185, 1, 2)
    tag = 0
    for i, con in enumerate(contours):
        # print(con.shape)
        if con.shape[0] == 185 :
            if tag == 0:
                tag = 1
                dots = np.squeeze(con)
                # cv2.drawContours(img, con, -1, (135, 210, 255), 5)  # 参数：图像、轮廓、轮廓序号（负数就画出全部轮廓）、颜色、粗细
                # cv2.imshow('a',img)
                # cv2.waitKey(0)
                for dot in dots:
                    if dot[0] != 75 and dot[0] != 74 and dot[1] != 87 and dot[1] != 86:
                        x = dot[0]
                        y = dot[1]
                        edge_dots.append([y,x-2])    # src在该点应该为0
            elif tag == 1:
                dots = np.squeeze(con)
                for dot in dots:
                    if dot[0] != 75 and dot[0] != 74  and dot[1] != 0  and dot[1] != 1:
                        x = dot[0]
                        y = dot[1]
                        edge_dots.append([y, x-2])  # src在该点应该为0
        edge_dots_np = np.array(edge_dots)
        np.save(left_triangle_npy, edge_dots_np)  #



def judgeCon(listdir, mode, formal_triangle_npy,left_triangle_npy):

    for img_name in listdir:
        # print(img_name)
        if mode:
            img_path = str(ori_path) + str(img_name)
        else:
            img_path = img_name

        result_code = 0  # 0镂空 ；1非镂空
        src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 1 3 6 10
        H = src.shape[0]
        W = src.shape[1]
        if W > H:       # 正 ： W87 * H75
            base_edges = np.load(formal_triangle_npy)
            # print('正')
            if W != 87 or H != 75:
                # print("尺寸不规格，非镂空")
                if mode:
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected'))
                else:
                    return "局部"
                continue
            for x_bottom in range(0,W):
                value = src[H-1, x_bottom]
                if value[0] !=0 :
                    result_code = 1
                    # print(str(img_name) + " 非镂空")
                    if mode:
                        os.system(
                        'cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected'))
                    else:
                        return "局部"
                    break

            if result_code == 1:
                continue

            for dot in base_edges:
                value = src[dot[0],dot[1]]
                if value[0] != 0:       # 左右边框上的点不是黑色
                    result_code = 1
                    # print(str(img_name) + " 非镂空")
                    if mode:
                        os.system(
                        'cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected'))
                    else :
                        return "局部"
                    break

            if not result_code:
                # print(str(img_name) + ' 镂空')
                if mode:
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/connected'))
                else:
                    return "镂空"





        else:           # 左 ： W75 * H87
            left_edges = np.load(left_triangle_npy)
            # print('左')
            # print(W)
            # print(H)
            if W not in [74,75,76] or H not in [86,87,88] :
                # print("尺寸不规格，非镂空")
                if mode:
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected'))
                else:
                    return "局部"
                continue

            for y_left in range(0, H):
                value = src[y_left, 1]
                if value[0] != 0:
                    result_code = 1
                    # print(str(img_name) + " 非镂空")
                    if mode:
                        os.system(
                        'cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected'))
                    else:
                        return "局部"
                    break
            if result_code == 1:
                continue

            for dot in left_edges:
                value = src[dot[0], dot[1]]
                if value[0] != 0:  # 上下边框上的点不是黑色
                    result_code = 1
                    # print(str(img_name) + " 非镂空")
                    if mode:
                        os.system(
                        'cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected'))
                    else:
                        return "局部"
                    break

            if not result_code:
                # print(str(img_name) + ' 镂空')
                if mode:
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/connected'))
                else:
                    return "镂空"
      #   save_formal_base_npy(src) # 测试时不运行。  保存正三角形的左右八字轮廓点集
      #   save_left_base_npy(src) # 测试时不运行。  保存正三角形的左右八字轮廓点集


if __name__ == "__main__":
    listdir = os.listdir(ori_path)
    # listdir = ['319019.png']
    # listdir = ['318514.png']     # 基线  (20, 1, 2)  (114, 1, 2)  (185, 1, 2)  (185, 1, 2)
    # listdir = ['319630.png']
    # listdir = ['319024.png']        # 左，基线
    # listdir = ['319630.png']        # 左
    mode = 1     # 0 演示单图  1 多图分批
    if mode:
        formal_triangle_npy = "../data/connectivity/saved/formal_triangle.npy"
        left_triangle_npy = "../data/connectivity/saved/left_triangle.npy"
    else:
        formal_triangle_npy = "../../data/connectivity/saved/formal_triangle.npy"
        left_triangle_npy = "../../data/connectivity/saved/left_triangle.npy"

    judgeCon(listdir, mode, formal_triangle_npy,left_triangle_npy )






