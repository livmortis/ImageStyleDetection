import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy.cluster.hierarchy as sch
import sklearn.cluster as sc

ori_path = "../data/testsym/ori/"
# ori_path = "../data/selftest/"
# ori_path = "../data/array/false_negative/"
# ori_path = "../data/array/disarrayZero/"
# ori_path = "../data/array/arrayZero/"
test_show = False
test_draw = False

T1 = 0.6    # 最近比第二近
# T1 = 0.2    # 最近比第五近
T2 = 2.3
def alpha_bg_to_white(img):
    B_channel = img[:,:,0]
    G_channel = img[:,:,1]
    R_channel = img[:,:,2]
    A_channel = img[:,:,3]
    b_eq_g = np.sum(B_channel) == np.sum(G_channel)
    b_eq_r = np.sum(B_channel) == np.sum(R_channel)

    # 画布改为白色 （画布就是透明的背景部分，也就是alpha等于0的部分，默认这部分的RGB也为0，是“透明的黑色”，改为“透明的白色”）'
    if test_show:
        # print(b_eq_r)
        # print(b_eq_g)
        print(np.sum(B_channel))
        print(np.sum(G_channel))
        print(np.sum(R_channel))
        # print(B_channel[0])
        print(np.sum(A_channel))
        # print(A_channel[0])
    # if np.sum(B_channel) == 0:
    #     # print("yez")
    #     B_channel[A_channel == 0] = 255
    #     G_channel[A_channel == 0] = 255
    #     R_channel[A_channel == 0] = 255
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


    # if b_eq_g and b_eq_r:
    #     # print('fuzz')
    #     img = cv2.blur(img, (3, 3), 10)         # 高斯滤波，解决了一些完整轮廓被检测的支离破碎的问题。
    #                     #适合纯黑的已被割裂的圆形轮廓， 不适合一些细节比较近似的图。
    #                     阵列不需要！
    return img


def manhattanDist(dot1, dot2):
    dist = np.sum(np.abs(dot1 - dot2))
    return dist

import config as cf

def classifyArray(listdir, mode, gy):
    if gy != -1:
        if gy == cf.XTYS or gy == cf.CF or gy == cf.DCX or gy == cf.ZW  :
            return '阵列'


    sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()

    for img_name in listdir:
        if mode:
            img_path = str(ori_path) + str(img_name)
        else:
            img_path = img_name

        print(img_path)
        src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 1 3 6 10
        try:
            imgshape = src.shape
        except:
            print(str(img_path)+"文件出错")
            if mode:
                os.system('cp %s %s'%(img_path,'../data/array/error' ))
                continue
            else:
                return "非阵列"

        # src = cv2.resize(src, (512, int(512 * (imgshape[0]/imgshape[1])) ))
        if test_draw:
            cv2.imshow('aa', src)
            cv2.waitKey(0)

        img = alpha_bg_to_white(src)
        if test_draw:
            cv2.imshow('bb', img)
            cv2.waitKey(0)

        # 1、灰度变换
        # img = cv2.Canny(img, 100, 200)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        if test_draw:
            cv2.imshow('cc', img)
            cv2.waitKey(0)
            # print(img[100])
        # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY )

        if test_draw:
            cv2.imshow('dd', img)
            cv2.waitKey(0)

        # 2、sift特征检测
        keypoints, descriptors = sift.detectAndCompute(img,None)
        # keypoints, descriptors = surf.detectAndCompute(img,None)
        keypoints_np = np.array(keypoints)
        # print(keypoints_np.shape)

        if test_draw:
            img = cv2.drawKeypoints(image=img, keypoints= keypoints, outImage=img, color=(51,163,236), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('a', img)
            cv2.waitKey(0)

        descriptors_np = np.array(descriptors)
        # print(descriptors_np.shape)



        # # # 3、遍历计算两两特征点(描述子)之间距离，通过此筛选高质量匹配点。
        filtered_descriptors_indexes = []
        if None in descriptors_np :
            if mode:
                print("没检测到关键点")
                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/array/error'))
            else:
                return "非阵列"
        else:
            for i, dot1 in enumerate(descriptors_np):
                dot1_dists_dict = {}
                for j, dot2 in enumerate(descriptors_np[i+1:]):
                    dot1_idx = i        # dot1的索引
                    dot2_idx = i+j+1    # dot2的索引
                    dot1_idx_and_dot2_idx = str(dot1_idx)+','+str(dot2_idx)     # 如 ‘0,1’

                    dist = manhattanDist(dot1, dot2)

                    dot1_dists_dict[dot1_idx_and_dot2_idx] = dist

                # print(dot1_dists_dict)           # 形如 {'0,1': 907.0, '0,2': 1408.0 ... '0,163': 1983.0}
                sorted_dists = sorted(dot1_dists_dict.items(), key= lambda xzy : xzy[1])
                # print(sorted_dists)             #  形如 [('0,127', 829.0), ('0,1', 907.0), ('0,146', 951.0)...  ('0,105', 4644.0)]
                if len(sorted_dists) > 1:
                    dot1_shotest_dist = sorted_dists[0]         # ('0,127', 829.0)      # 最近点
                    dot1_secondShort_dist = sorted_dists[1]     #  ('0,1', 907.0)     # 第二近的点
                    # if len(sorted_dists) < 6:
                    #     dot1_fifthShort_dist = sorted_dists[-1]
                    # else:
                    #     dot1_fifthShort_dist = sorted_dists[5]                            # 第五近的点


                    ratio = dot1_shotest_dist[1]/dot1_secondShort_dist[1]     # 829/907     # 1/2比值
                    # ratio = dot1_shotest_dist[1]/dot1_fifthShort_dist[1]                  # 1/5比值


                    if  ratio < T1 :
                        dot1_idx_decode = str(dot1_shotest_dist[0]).split(',')[0]       # 0    点1的索引
                        dot2_idx_decode = str(dot1_shotest_dist[0]).split(',')[1]     # 127     离点1最近的点2的索引
                        # dot3_idx_decode = str(dot1_secondShort_dist[0]).split(',')[1]   # 1     离点1第二近的点3的索引

                        filtered_descriptors_indexes.append(dot1_idx_decode)  # 保存本身点
                        filtered_descriptors_indexes.append(dot2_idx_decode)    # 保存最近点
                        # filtered_descriptors_indexes.append(dot3_idx_decode)    # 应该不保存第二近的点 ？？？


            des = []

            for idx in filtered_descriptors_indexes:
                des.append(descriptors_np[int(idx)])
            des = np.array(des)

            if des.shape[0] <=2:    # 只有一个无法比较距离，下方会报错。
                print('阵列，sift关键点小于2')
                if mode:
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/array/disarray64'))
                    continue
                else:
                    return "非阵列"



            disMat = sch.distance.pdist(des, 'euclidean')
            # disMat = sch.distance.pdist(descriptors_np, 'euclidean')  # 不采用第三步。

            Z = sch.linkage(disMat, method='average')
            # sch.dendrogram(Z, leaf_rotation=90, leaf_font_size=6,)
            # plt.show()

            # # cluster = fcluster(Z, t=1, criterion='inconsistent')
            # cluster = sch.fcluster(Z, t=20, criterion='distance')

            z_dists = Z[:,2]
            if(test_show):
                print(Z)
            # print(z_dists)

            # a、计算距离100以内的匹配点的个数
            # small_than_100_num = len(z_dists[z_dists<100])
            # print(small_than_100_num)

            # b、计算中位数
            # z_median = np.median(z_dists)
            # print(z_median)
            #
            # c、计算0的个数
            z_zeron_list =z_dists[z_dists==0]
            zero_num = len(z_zeron_list)
            # if test_show:
            print('0的个数：'+str(zero_num))

            # d、0的个数除以总个数
            total_len = len(z_dists)
            ratio = zero_num/total_len
            if test_show:
                print("目前策略，0的比例："+str(ratio))
            # if (zero_num>40 and ratio > 0.45) or zero_num > 100:  # 保存了第二近的点
            # if (zero_num>15 and ratio > 0.2) or zero_num > 50:     # 没保存第二近的点，仅保存本身点和最近点。 比值0.6. 优
            # if (zero_num>20 and ratio > 0.3) or zero_num > 50:     # 没保存第二近的点，仅保存本身点和最近点。比值改为0.7。 第二优
            if (zero_num>10  and ratio > 0.2) or zero_num > 20:     # 没保存第二近的点，仅保存本身点和最近点。比值改为0.7。 第二优
            # if (zero_num>10 and ratio > 0.2) or zero_num > 40:     # 没保存第二近的点，仅保存本身点和最近点。比值改为0.5。
            # if (zero_num>15 and ratio > 0.25) or zero_num > 50:     # 没保存第二近的点，仅保存本身点和最近点. 比值改为大于0.8。 差
            # if (zero_num>8 and ratio > 0.2) or zero_num > 50:     # 没保存第二近的点，没保存本身点，仅保存最近点
            # if (zero_num>80 and ratio > 0.40) or zero_num > 200:     # 没有判断一二点比值
            # if (zero_num>=5 and ratio > 0.25) or zero_num > 50:     # 判断一五点比值，仅保存最近点
                if mode:
                    print("阵列")

                    # os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/array/arrayZero'))
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/array/array64'))
                else:
                    return "阵列"

            else:
                if mode:
                    print("非阵列")

                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/array/disarray64'))
                else:
                    return "非阵列"

            # e、小于20的个数除以总个数
            # small_than_20_num = len(z_dists[z_dists<20])
            # ratio2 = small_than_20_num/total_len
            # print(ratio2)
            # if zero_num>35 and ratio2 > 0.6:
            #     os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/array/array'))
            # else:
            #     os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/array/disarray'))



if __name__ == "__main__":
    listdir = os.listdir(ori_path)
    mode = 1  # 0 演示单图  1 多图分批
    if test_show:
        # listdir = ['10009.png']
        # listdir = ['10379.png']   # 典型阵列，却一直分错。
        # listdir = ['10297.png']   # 典型阵列，却一直分错，树。
        # listdir = ['10320.png']     # 透明底黑前景 狗
        # listdir = ['10650.png']     # 透明底黑前景 狗
        listdir = ['10520.png']     # 新图
        # listdir = ['10110.png']

    classifyArray(listdir, mode)


















