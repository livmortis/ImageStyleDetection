import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import config as cf


# 2020.6.2  余弦，阈值0.1


ori_path = "../data/testsym/ori/"
TEST = False
TRESHOD_TYPE = 2    # 0固定， 1自适应， 2不适用阈值
DIST_ALGO = 1  # 0 欧式距离  1 余弦相似度
np.set_printoptions(threshold=np.inf)

def alpha_bg_to_white(img):
    B_channel = img[:,:,0]
    G_channel = img[:,:,1]
    R_channel = img[:,:,2]
    A_channel = img[:,:,3]
    # print(np.sum([B_channel, G_channel, R_channel]) )
    # print(B_channel)
    # print(np.sum(B_channel))
    # print(np.sum(G_channel))
    # print(np.sum(R_channel))
    # print(np.sum(A_channel))

    # BGR三通道都为黑色，alpha通道控制图像形状
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


    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if TRESHOD_TYPE == 0 :
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        # 固定阈值
    elif TRESHOD_TYPE == 1:
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.ADAPTIVE_THRESH_MEAN_C, 99 , 2)
        # 自适应阈值，分错的：1001.png、1000.png、10634.png、10023.png等左右颜色不同

    return img



def compute_cosin_distance(vec1, vec2):
    vec1 = vec1 / (np.max(vec1) + 0.01)
    vec2 = vec2 / (np.max(vec2) + 0.01)
    dist = 1 - np.dot(vec1, vec2)/ (np.linalg.norm(vec1)* np.linalg.norm(vec2)+0.01)
    return dist

def eucliDist(A,B):
    return np.sqrt(sum(np.power((A - B), 2)))


def hist_method(img_name,img,H,W):

    # 左右分割
    half_left = img[:, 0:int(W / 2)]
    if W % 2 == 0:
        half_right = img[:, int(W / 2):W]
    else:
        half_right = img[:, int(W / 2) + 1:W]

    # 上下分割后，左右分割
    if H % 2 == 0:
        H_half = int(H / 2)
    else:
        H_half = int(H / 2) + 1
    up_left = half_left[:int(H / 2), :]
    down_left = half_left[H_half:H, :]
    up_right = half_right[:int(H / 2), :]
    down_right = half_right[H_half:H, :]

    if TEST:
        cv2.imshow('half_left', half_left)
        cv2.imshow('half_right', half_right)
        cv2.imshow('up_left', up_left)
        cv2.imshow('up_right', up_right)
        cv2.imshow('down_left', down_left)
        cv2.imshow('down_right', down_right)
        cv2.waitKey(0)

    # 左与右
    left_hist = cv2.calcHist([half_left], [0], None, [256], [0, 256])
    right_hist = cv2.calcHist([half_right], [0], None, [256], [0, 256])
    # print(left_hist)
    # print(right_hist)
    dist_cor = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)
    # cv2.HISTCMP_CORREL: 皮尔逊相关系数，1最高 ;    cv2.HISTCMP_CHISQR: 卡方检验，0最高
    # cv2.HISTCMP_INTERSECT: 十字交叉性,0最高;   cv2.HISTCMP_BHATTACHARYYA: 巴氏距离，1最高

    # 左上与右上
    up_left_hist = cv2.calcHist([up_left], [0], None, [256], [0, 256])
    up_right_hist = cv2.calcHist([up_right], [0], None, [256], [0, 256])
    dist_cor_up = cv2.compareHist(up_left_hist, up_right_hist, cv2.HISTCMP_CORREL)

    # 左下与右下
    down_left_hist = cv2.calcHist([down_left], [0], None, [256], [0, 256])
    down_right_hist = cv2.calcHist([down_right], [0], None, [256], [0, 256])
    dist_cor_down = cv2.compareHist(down_left_hist, down_right_hist, cv2.HISTCMP_CORREL)
    mean_score = np.mean([dist_cor, dist_cor_up, dist_cor_down])

    if TEST:
        print(dist_cor)
        print(dist_cor_up)
        print(dist_cor_down)
        print(mean_score)
        print('\n')
    else:
        if mean_score > 0.9999:
            if mode:
                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/testsym/symm_by_cor'))
            else:
                return "对称"
        else:
            if mode:
                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/testsym/nosymm_by_cor'))
            else:
                return "不对称"






def pixel_cosin_method(mode, img_name,img):     # 余弦距离和欧氏距离
    rota90_img = np.zeros([img.shape[1], img.shape[0]])
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            a = img[y, x]
            rota90_img[x, y] = a

    multi_angle_imgs = [img, rota90_img ]
    dists_list_with_multi_angle = []
    for i, theImg in enumerate(multi_angle_imgs):
        W = theImg.shape[1]
        # 左右分割
        half_left = theImg[:, 0:int(W / 2)]
        if W % 2 == 0:
            half_right = theImg[:, int(W / 2):W]
        else:
            half_right = theImg[:, int(W / 2) + 1:W]

        # 右半部分逆序
        new_right = []
        for row in half_right:
            new_row = row[::-1]
            new_right.append(new_row)
        new_right = np.array(new_right)
        if TEST:
            cv2.imshow("a",half_left)
            cv2.imshow("b",new_right)
            cv2.waitKey(0)

        '''1、sift特征点匹配法, 弃用'''
        # # sift左右分别检测特征点
        # sift = cv2.xfeatures2d.SIFT_create()
        # l_kpt, l_dscp = sift.detectAndCompute(half_left, None)
        # r_kpt, r_dscp = sift.detectAndCompute(new_right, None)


        # # 匹配描述子
        # bfmatcher = cv2.BFMatcher()
        # dmatches = bfmatcher.match(l_dscp,r_dscp)
        # dists = []
        # for d in dmatches:
        #     dists.append(d.distance)
        # dist_mean = np.mean(dists)

        # if dist_mean<250:
        #     os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/testsym/symm_by_cor'))
        # else:
        #     os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/testsym/nosymm_by_cor'))

        '''2、直接求像素值余弦距离法'''
        dists = []
        for l, r in zip(half_left,new_right):
            # print(l)
            # print(r)
            dist = compute_cosin_distance(l,r) if DIST_ALGO else eucliDist(l,r)
            dists.append(dist)
        dists = np.array(dists)
        dists_mean = np.mean(dists)

        if TEST:
            print('余弦距离：'+str(dists_mean)) if DIST_ALGO else  print('欧式距离：'+str(dists_mean))


        threshold = 0.1 if DIST_ALGO else 30       # 余弦距离阈值0.1, 欧氏距离阈值30
        if dists_mean < threshold:
            # print("对称")
            if mode:
                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/testsym/symm_by_cor'))
            else:
                return "对称"
            break
        else:
            if i==1:
                # print("不对称")
                if mode:
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/testsym/nosymm_by_cor'))
                else:
                    return "不对称"
            else:
                continue





def judgeSym(listdir, mode, gy, gyid):
    if gy != -1:
        if gy == cf.XTYS  or gy == cf.CF:
            return "对称"
        elif gy == cf.TD:
            if gyid in [8,9,10,11] or gyid in [26,27,28,29] or gyid in [44,45,46,47] :
                return "对称"

    for img_name in listdir:
        if mode:
            img_path = str(ori_path) + str(img_name)
        else:
            img_path = img_name
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)       # 1 3 6 10
        imgshape = img.shape
        # img = cv2.resize(img, (512, int(512 * (imgshape[0]/imgshape[1])) ))
        # img = cv2.imread(str(ori_path)+str('10222.png'),  cv2.IMREAD_UNCHANGED)
        if TEST:
            cv2.imshow('img before abtw', img)
            cv2.waitKey(0)
        img = alpha_bg_to_white(img)
        if TEST:
            cv2.imshow('img after abtw', img)
            cv2.waitKey(0)
        # print('\n')
        # print(img_name)
        # print(img.shape)

        # hist_method(img_name,img,H,W)                    # 直方图法
        result = pixel_cosin_method(mode, img_name,img)     # 距离法

    return result

if __name__ == "__main__":
    listdir = os.listdir(ori_path)
    mode = 1     # 0 演示单图  1 多图分批
    if TEST:
        # listdir = ['10389.png' , '10525.png', '10235.png', '10085.png']
        # listdir = ['10206.png']
        # listdir = ['10253.png']
        # listdir = ['10320.png']     # 狗，应该不对称， “透明底黑前景图”典型
        # listdir = ['10987.png']
        # listdir = ['379253.png']  # 应该对成
        # listdir = ['379253.png']
        # listdir = ['379278.png']    # 应该不对称， “透明底黑白前景图”典型
        # listdir = ['11014.png']         # 不对称高跟鞋    余弦0.11  欧式1723
        # listdir = ['10527.png']              # 不对称大象   余弦0.47    欧式32
        # listdir = ['10335.png']              # 不对称狐狸   余弦0.37   欧式34
        # listdir = ['10295.png']                # 不对称松鼠   余弦0.56    欧式29
        # listdir = ['10901.png']               # 完全不对称小鹿   余弦0.21    欧式25
        # listdir = ['10452.png']               # 稍微不对称海马   余弦0.098    欧式25       放弃欧氏距离
        listdir = ['236852.png']
    judgeSym(listdir, mode)