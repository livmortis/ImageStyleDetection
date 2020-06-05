
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import config as cf

# ori_path = "../data/testsym/ori/"
ori_path = "../data/connectivity/ori/"
test_show = False
def alpha_bg_to_white(img):
    B_channel = img[:,:,0]
    G_channel = img[:,:,1]
    R_channel = img[:,:,2]
    A_channel = img[:,:,3]
    b_eq_g = np.sum(B_channel) == np.sum(G_channel)
    b_eq_r = np.sum(B_channel) == np.sum(R_channel)


    # 画布改为白色 （画布就是透明的背景部分，也就是alpha等于0的部分，默认这部分的RGB也为0，是“透明的黑色”，改为“透明的白色”）
    # if np.sum(B_channel) == 0 :
    #     # print("yez")
    #     B_channel[ A_channel == 0 ] = 255
    #     G_channel[ A_channel == 0 ] = 255
    #     R_channel[ A_channel == 0 ] = 255
    #     ret, img = cv2.threshold(img, thresh=254, maxval=255, type=cv2.THRESH_BINARY_INV)

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
        ret, img = cv2.threshold(img, thresh=254, maxval=255, type=cv2.THRESH_BINARY_INV)

    if b_eq_g and b_eq_r:
        # print('fuz')
        img = cv2.blur(img, (3, 3), 10)

    return img


# 去掉最外层轮廓 , 弃用 -- 图像内容可能填满画布
# def delBiggestCont(contours, w, h):
#     newCons = []
#     for cont in contours:
#         '''轮廓外接矩形'''
#         rect = cv2.minAreaRect(cont)
#         rectW = rect[1][0]
#         rectH = rect[1][1]
#
#         if rectW > w * 0.8 and rectH > h * 0.8 :
#             # print("rectWith: " + str(rectWith))
#             # print("\nWIDTH: " + str(WIDTH))
#             # print("\nrectLen: " + str(rectLen))
#             # print("\nLENGTH: " + str(LENGTH))
#             continue
#         else:
#             newCons.append(cont)




'''弃用'''
def houghLineRound(img, img_name, H, W):
    houghLin = cv2.HoughLinesP(img, rho=0.2, theta=np.pi / 50, threshold=30, lines=20, minLineLength=50,
                               maxLineGap=20)
    #  rho正比      theta反比        反比                         正比             正比
    if test_show:
        print("全量霍夫直线：" + str(houghLin))

    if True in np.squeeze(houghLin == None):
        # print('结论：严重圆形轮廓')
        if mode :
            os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/shape_round'))
        else:
            return "弧形轮廓"
    else:
        if not houghLin.shape[0] == 1:
            houghLin = np.squeeze(houghLin)
        else:
            houghLin = np.squeeze(houghLin)
            houghLin = [houghLin]

        # print('有直线')
        line_counts = 0

        for line in houghLin:
            x1, y1, x2, y2 = line
            if (x1 == 0 and x2 == 0) or \
                    (y1 == 0 and y2 == 0) or \
                    ((x1 in list(range(W - 5, W))) and (x2 == x1)) or \
                    ((y1 in list(range(H - 5, H))) and (y2 == y1)):
                continue
            else:
                line_counts += 1

        if line_counts < 10:
            # print('直线数量小于2：' + str(line_counts))
            # print('结论：圆形轮廓')
            if mode:
                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/shape_round'))
            else:
                return "弧形轮廓"
        else:
            # print('结论：含角轮廓')
            if mode:
                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/shape_angle'))
            else:
                return "角轮廓"
            # print('直线数量：' + str(line_counts))
        empty_img2 = np.zeros([H, W, 4])

        for line in houghLin:
            x1, y1, x2, y2 = line
            cv2.line(empty_img2, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
        if test_show:
            cv2.imshow('bbb', empty_img2)
            cv2.waitKey(0)



'''检测圆弧轮廓'''
def houghCircleRound(img, img_name, H, W):
    short_edge = np.min([H, W])
    big_edge = np.max([H, W])
    circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=2, minDist=100, param1=50, param2=80,
                               minRadius=int(short_edge * 0.2), maxRadius=int(big_edge * 5))

    if circles is None:
        # print('未检测到圆')
        if mode:
            os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/shape_angle'))
        else:
            return "角轮廓"
        # exit(-1)
    else:
        if mode:
            os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/shape_round'))
        else:
            return "弧形轮廓"
        if test_show:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 255), 2)
                # cv2.circle(img, (i[0], i[1]), 2, (255, 255, 255), 3)
                cv2.imshow('detected chess', img)
                cv2.waitKey(0)


'''检测矩形轮廓, 外接矩形/凸包'''
def surroundContour(img, img_name, biggest_cont, H, W):

    '''轮廓凸包'''
    # contours = np.array(contours)
    # contours = np.squeeze(contours)


    hull = cv2.convexHull(biggest_cont)
    hull = np.array(hull)
    last_x = -1
    last_y = -1
    rect_edge_num = 0
    for dot in hull:
        dot = np.squeeze(dot)
        x = dot[0]
        y = dot[1]
        if (x == last_x and np.abs(y-last_y) > H*0.25) or\
                (y == last_y and np.abs(x-last_x) > W*0.25):
            rect_edge_num += 1
        last_x = x
        last_y = y

    # 绘制凸包
    length = len(hull)
    for i in range(len(hull)):
        cv2.line(img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (255, 255, 255), 2)
    cv2.imshow('line', img)
    cv2.waitKey()
    if rect_edge_num >= 3:
        # print("是矩阵，矩阵边数："+str(rect_edge_num))
        if mode:
            os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/shape_cube'))
        else:
            return "方正轮廓"
    else:
        if mode:
            os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/nocube'))
        else:
            return "666"



    '''轮廓外接矩形'''
    # rect = cv2.minAreaRect(contours)
    # angle = rect[2]

    # if angle==0:
    #     os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/shape_cube'))





def clasConShape(listdir, mode, gy, gyid):
    if gy != -1:
        if gy == cf.SJX  :
            return '角轮廓'
        elif gy == cf.GZ:
            if gyid in [30,33,36,39]:
                return "方正轮廓"
            else:
                return "弧形轮廓"
        elif gy == cf.PZ:
            if gyid <200:
                return "弧形轮廓"
            elif gyid>=200 and gyid<300:
                return "方正轮廓"


    for img_name in listdir:
        if mode:
            img_path = str(ori_path) + str(img_name)
        else:
            img_path = img_name

        src = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)       # 1 3 6 10
        imgshape = src.shape
        # src = cv2.resize(src, (512, int(512 * (imgshape[0]/imgshape[1])) ))
        # print(src.shape)
        H = src.shape[0]
        W = src.shape[1]
        if test_show:
            cv2.imshow('before white',src)
            cv2.waitKey(0)
        img = alpha_bg_to_white(src)
        if test_show:
            cv2.imshow('after white',img)
            cv2.waitKey(0)
        # b, g, r, a = cv2.split(src)
        # img = cv2.merge([r, g, b, a])
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        if test_show:
            cv2.imshow("src : ", img)
            cv2.waitKey(0)

        i, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE 所有边界点, RETR_LIST不建立等级关系
        # contours = delBiggestCont(contours,W,H)   # 弃用


        empty_img_allCons = np.zeros([H, W, 4], dtype=np.int8)
        cv2.drawContours(empty_img_allCons, contours, -1, (255, 180, 255), 1)  # 参数：图像、轮廓、轮廓序号（负数就画出全部轮廓）、颜色、粗细
        if test_show:
            cv2.imshow("origin contour : ", empty_img_allCons)
            cv2.waitKey(0)
        empty_img_allCons = empty_img_allCons.astype(np.uint8)
        ret, img_allCons = cv2.threshold(empty_img_allCons, 0, 255, cv2.THRESH_BINARY)
        img_allCons = cv2.Canny(img_allCons, 100, 200)

        '''找最大轮廓'''
        area = []
        for i in range(len(contours)):
            area.append(cv2.contourArea(contours[i]))
        max_idx = np.argmax(np.array(area))
        biggest_cont = contours[max_idx]


        '''将原图变为仅有一个外轮廓的图。'''
        empty_img_biggestCons = np.zeros([H, W, 4], dtype=np.int8)
        cv2.drawContours(empty_img_biggestCons, biggest_cont, -1, (255, 180, 255), 1)  # 参数：图像、轮廓、轮廓序号（负数就画出全部轮廓）、颜色、粗细
        if test_show:
            cv2.imshow("biggest contour : ", empty_img_biggestCons)
            cv2.waitKey(0)
        # empty_img_biggestCons = cv2.cvtColor(empty_img_biggestCons, cv2.COLOR_RGB2GRAY)
        empty_img_biggestCons = empty_img_biggestCons.astype(np.uint8)
        ret, img = cv2.threshold(empty_img_biggestCons, 0, 255, cv2.THRESH_BINARY)
        img = cv2.Canny(img, 100, 200)


        # surroundContour(img,img_name,  biggest_cont, H, W)    # 凸包检测矩形轮廓
        # houghLineRound(img, img_name, H, W)      # 霍夫直线法检测圆弧轮廓，弃用
        # houghCircleRound(img, img_name, H, W)            # 霍夫圆法检测圆弧轮廓


        if test_show:
            cv2.imshow("threshold and canny: ", img)
            cv2.waitKey(0)



        '''    flow    '''
        # 1、 凸包检测矩形轮廓
        hull = cv2.convexHull(biggest_cont)
        hull = np.array(hull)
        last_x = -1
        last_y = -1
        rect_edge_num = 0
        for dot in hull:
            dot = np.squeeze(dot)
            x = dot[0]
            y = dot[1]
            if (x == last_x and np.abs(y - last_y) > H * 0.5) or \
                    (y == last_y and np.abs(x - last_x) > W * 0.5):
                rect_edge_num += 1
            last_x = x
            last_y = y

        # 绘制凸包
        # length = len(hull)
        # for i in range(len(hull)):
        #     cv2.line(img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (255, 255, 255), 2)
        # cv2.imshow('line', img)
        # cv2.waitKey()
        if rect_edge_num >= 3:
            # print("是矩阵，矩阵边数：" + str(rect_edge_num))
            if mode:
                os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/shape_cube'))
            else:
                return "方正轮廓"
        # 2、检测圆弧和角轮廓
        else:
            short_edge = np.min([H, W])
            big_edge = np.max([H, W])
            circles = cv2.HoughCircles(img_allCons, method=cv2.HOUGH_GRADIENT, dp=2.5, minDist=100, param1=50, param2=80,
                                       minRadius=int(short_edge * 0.2), maxRadius=int(big_edge * 5))

            if circles is None:
                # print('是角度轮廓图')
                if mode:
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/shape_angle'))
                else:
                    return "角轮廓"
                # exit(-1)
            else:
                # print('是弧形轮廓图')
                if mode :
                    os.system('cp %s %s' % (str(ori_path) + str(img_name), '../data/contourShape/shape_round'))
                else:
                    return "弧形轮廓"
                if test_show:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        cv2.circle(img_allCons, (i[0], i[1]), i[2], (255, 255, 255), 2)
                        # cv2.circle(img_allCons, (i[0], i[1]), 2, (255, 255, 255), 3)
                        cv2.imshow('detected chess', img_allCons)
                        cv2.waitKey(0)



if __name__ == "__main__":
    listdir = os.listdir(ori_path)
    mode = 1     # 0 演示单图  1 多图分批
    if test_show:
        # listdir = ['377925.png']
        listdir = ['367640.png']
    clasConShape(listdir, mode, -1)
