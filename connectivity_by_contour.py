# 圆和方碰撞元素

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from utils.exception import ContourShapeDectectException

ori_path = "../data/connectivity/ori/"

limit_dot = np.array([0,1,2,3,4,5,512,511,510,509,508])
np.set_printoptions(threshold=np.inf)


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


# 圆形原始元素的外轮廓，存储npy。
def save_round_np(contours):
    con_dots = []
    for con in contours:  # 循环4次，每个是四个角的每一个轮廓
        con_seq = np.squeeze(con)
        for con_dot in con_seq:
            if 511 in con_dot or 0 in con_dot:  # 循环轮廓数次，
                continue
            else:
                con_dots.append(con_dot)
    con_dots = np.array(con_dots)
    con_extend = np.concatenate([con_dots, con_dots + 5, con_dots - 5])         # 2000变6000
    con_str = []
    for one_pair in con_extend:
        con_str.append(str(one_pair))       # 变为string，因为np.intersect1d不支持二维求交集
    con_str_np = np.array(con_str)
    np.save(round_ori, con_str_np)

# 圆方形原始元素的外轮廓，存储npy。
def save_rouCube_np(contours):
    con_dots = []
    for con in contours:
        if con.shape[0] == 601:
            continue                     # 这是样本图1797_110.png本身多余的一块轮廓。该轮廓601个点，其他四个角轮廓210个点。
        con_seq = np.squeeze(con)
        for con_dot in con_seq:
            if 511 in con_dot or 0 in con_dot:
                continue
            else:
                con_dots.append(con_dot)
    con_dots = np.array(con_dots)
    con_extend = np.concatenate([con_dots, con_dots + 5, con_dots - 5])     # 300变900
    con_str = []
    for one_pair in con_extend:
        con_str.append(str(one_pair))
    con_str_np = np.array(con_str)
    np.save(rouCube_ori, con_str_np)



def draw_contours(img, contours):
    cv2.drawContours(img, contours, -1, (135, 210, 255), 5)  # 参数：图像、轮廓、轮廓序号（负数就画出全部轮廓）、颜色、粗细
    cv2.imshow("7_contour", img)
    cv2.moveWindow("7_contour", 300, 10)
    cv2.waitKey(0)

def judgeCon(listdir, mode, round_np,rouCube_np):
    try:
        for img_name in listdir:
            if mode:
                img_path = str(ori_path) + str(img_name)
            else:
                img_path = img_name

            # print(img_path)
            src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)       # 1 3 6 10
            imgshape = src.shape
            # src = cv2.resize(src, (512, int(512 * (imgshape[0]/imgshape[1])) ))
            img = alpha_bg_to_white(src)
            b, g, r, a = cv2.split(img)
            img = cv2.merge([r, g, b, a])
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            W = img.shape[1]
            H = img.shape[0]
            # print(img.shape)


            i, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE 所有边界点, RETR_LIST不建立等级关系
            contours = np.array(contours)



            # draw_contours(src, contours)      # 测试不执行
            # save_round_np(contours)            # 测试不执行
            # save_rouCube_np(contours)          # 测试不执行



            four_conner = [img[5, 5], img[H - 5, 5], img[5, W - 5], img[H - 5, W - 5]]
            four_conner_shrank = [img[40, 40], img[H - 40, 40], img[40, W - 40], img[H - 40, W - 40]]

            if 0 in four_conner:  # 方形
                # print('方形')
                # print(img_name)
                # print(img.shape)
                all_contours = []
                for con in contours:
                    con_flat = con.flatten()
                    all_contours.extend(con_flat)
                all_contours_np = np.array(all_contours)

                intersection = np.intersect1d(all_contours_np, limit_dot)
                if len(intersection) == 0:
                    # print(str(img_name) + "镂空")
                    if mode:
                        # print(str(img_name) + "镂空")
                        os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/connected1'))
                    else:
                        return "镂空"
                else:
                    # print(str(img_name) + "局部")
                    if mode :
                        os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected1'))
                    else:
                        return "局部"

            elif 0 in four_conner_shrank:  # 圆方形
                # print('圆方形')
                all_contours_str = []
                for one_con in contours:
                    for one_dot in one_con:
                        one_dot = np.squeeze(one_dot)
                        # 右下 448
                        # 左下 63
                        if (one_dot[0] in [0,1] and one_dot[1] in list(range(75, 430))  ) or\
                            (one_dot[0] in  list(range(75, 430)) and one_dot[1] in [0,1]) or\
                            (one_dot[0] in  list(range(75, 430)) and one_dot[1] in [510,511,512]) or\
                            (one_dot[0] in [510,511,512] and one_dot[1] in  list(range(75, 430))):
                            # print(str(img_name) + "局部")
                            if mode:
                                os.system(
                                    'cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected1'))
                            else:
                                return "局部"
                            break
                        all_contours_str.append(str(one_dot))
                    else:
                        continue
                    break
                all_contours_np = np.array(all_contours_str)

                intersection = np.intersect1d(all_contours_np, rouCube_np)
                # print(len(intersection))

                if len(intersection) in [319, 320, 321, 322, 323]:
                    # print(str(img_name) + "镂空")
                    if mode:
                        os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/connected1'))
                    else:
                        return "镂空"
                else:
                    # print(str(img_name) + "局部")
                    if mode:
                        os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected1'))
                    else:
                        return "局部"

            else:  # 圆形
                # print('圆形')
                all_contours_str = []
                for one_con in contours:
                    for one_dot in one_con:
                        one_dot = np.squeeze(one_dot)
                        if (one_dot[0] in [0,1] and one_dot[1] in [255,256,257]) or\
                            (one_dot[0] in [255,256,257] and one_dot[1] in [0,1]) or\
                            (one_dot[0] in [255,256,257] and one_dot[1] in [510,511,512]) or\
                            (one_dot[0] in [510,511,512] and one_dot[1] in [255, 256, 257]) :

                            # print(str(img_name) + "贴边, 一定不镂空")
                            if mode:
                                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected1'))
                            else:
                                return "局部"
                            break
                        all_contours_str.append(str(one_dot))
                    else:
                        continue
                    break
                all_contours_np = np.array(all_contours_str)

                # print(all_contours_np)
                # print(round_np)
                intersection = np.intersect1d(all_contours_np, round_np)
                # print(len(intersection))

                if len(intersection) in [1225,1226,1227,1228,1229]:
                    # print(str(img_name) + "镂空")
                    if mode:
                        os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/connected1'))
                    else:
                        return "镂空"
                else:
                    # print(str(img_name) + "局部")
                    if mode:
                        os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/connectivity/disconnected1'))
                    else:
                        return "局部"
    except Exception as e:
        raise ContourShapeDectectException(str(e))


        # for cont in contours:
        #     '''轮廓外接矩形'''
        #     rect = cv2.minAreaRect(cont)
        #     rectWith = rect[1][0]
        #     rectLen = rect[1][1]
        #
        #     if rectWith > WIDTH * 0.5 and rectLen > LENGTH * 0.5 and rectWith < WIDTH * 0.9:
        #         # print("rectWith: " + str(rectWith))
        #         # print("\nWIDTH: " + str(WIDTH))
        #         # print("\nrectLen: " + str(rectLen))
        #         # print("\nLENGTH: " + str(LENGTH))
        #
        #         box = cv2.boxPoints(rect)
        #         box = np.int0(box)
        #
        #         for i in range(len(box)):
        #             d = cv2.line(aa, tuple(box[i]), tuple(box[(i + 1) % 4]), (0, 255, 255), 5, 2)
        #         if PREVIEW_EVERY_STEP:
        #             cv2.imshow("8_rect" + str(successNum), d)
        #             cv2.moveWindow("8_rect" + str(successNum), 300, 500)
        #             cv2.waitKey(0)
        #
        #         successNum += 1
        #
        # totalNum += 1




        # print(cs.shape)

if __name__ == "__main__":
    listdir = os.listdir(ori_path)
    mode = 1     # 0 演示单图  1 多图分批
    if mode:
        round_ori = "../data/connectivity/saved/round.npy"
        rouCube_ori = "../data/connectivity/saved/rouCube.npy"
    else:
        round_ori = "../../data/connectivity/saved/round.npy"
        rouCube_ori = "../../data/connectivity/saved/rouCube.npy"
    # listdir = [
    #             # '1245_299.png' ,  # 方
    #             '1797_110.png',  # 圆方
    #             # '90_0.png',  # 圆
    #             ]
    # listdir = ['85_12.png'] #镂空圆
    # listdir = ['111_53.png']
    # listdir = ['1780_154.png']      # 圆方的错误镂空
    # listdir = ['378843.png']
    # listdir = ['85_12.png']
    # listdir = ['378833.png']
    round_np = np.load(round_ori)
    rouCube_np = np.load(rouCube_ori)

    judgeCon(listdir, mode,  round_np,rouCube_np)


