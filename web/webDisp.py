#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@File    : webapp.py.py
@Author  : wangwei
@Time    : 2019/11/11 下午3:21
"""
from flask import Flask, request
import json
import os
from flowjob import flow_model
import numpy as np
round_ori = "../../data/connectivity/saved/round.npy"
rouCube_ori = "../../data/connectivity/saved/rouCube.npy"
formal_triangle_npy = "../../data/connectivity/saved/formal_triangle.npy"
left_triangle_npy = "../../data/connectivity/saved/left_triangle.npy"

app = Flask(__name__)

imgs = []
type = 0   # 0 互联网元素  1 碰撞元素  2 碰撞三角形
online = False

temppath = ["../../data/yfimg/"]
stpath = ["../../data/selftest/"]
rootpathes = [ "../../data/testsym/ori/", "../../data/connectivity/ori/", "../../data/connectivity/ori_triangle/"]

# imgnames = os.listdir(rootpathes[type])
# imgnames = os.listdir(temppath[type])
imgnames = os.listdir(stpath[type])

for i in imgnames:
    # imgs.append(rootpathes[type]+str(i))
    # imgs.append(temppath[type]+str(i))
    imgs.append(stpath[type]+str(i))

fm = flow_model()


@app.route('/test', methods=['GET'])
def test():
    if request.method == 'GET':
        ii = request.args.get('i', '0')
        ii = int(ii)
        # ii = 106
        img_path = imgs[ii]
        # img_path =  "../../data/testsym/ori/379253.png"       # 演示网络图片
        print(img_path)
        if not type:
            output = fm.inference_inet(img_path, -1)
        elif type == 1:
            output = fm.inference_pz1(img_path,round_ori,rouCube_ori)
        else:
            output = fm.inference_pz2(img_path, formal_triangle_npy,left_triangle_npy )

        print(output)

        if not type:
            rst = """<table border="1">
                        <tr>
                            <td> 复杂度 </td>
                            <td> %s </td>
                        </tr>
                        <tr>
                            <td> 轮廓形状 </td>
                            <td> %s </td>
                        </tr>
                        <tr>
                            <td> 对称性 </td>
                            <td> %s </td>
                        </tr>
                        <tr>
                            <td> 高或扁 </td>
                            <td> %s </td>
                        </tr>
                        <tr>
                            <td> 阵列 </td>
                            <td> %s </td>
                        </tr>
                    """ % ( output[0],output[1],output[2],output[3],output[4] )
        else:
            rst = """<table border="1">
                                   <tr>
                                       <td> 是否镂空 </td>
                                       <td> %s </td>
                                   </tr>
                               """ % (output[0])

        rst += """<a href="?i=%s">上一个</a>当前图片路径:%s <a href="?i=%s">下一个</a>""" % (ii - 1, img_path.split('/')[-1][:-4], ii + 1)
        if not type:
            rst += """<iframe src="https://oss.looodesign.com/test/img/%s.svg" width="300" height="300"></iframe>"""%(img_path.split('/')[-1][:-4])
        elif type ==1 :
            if not online:
                src_img_path = img_path.split('/')[-1][:-4].split('_')[0]
                gy_img_path = img_path.split('/')[-1][:-4].split('_')[1]
                print(src_img_path)
                print(gy_img_path)
                rst += """<iframe src="http://124.207.72.10:8000/api/svg?src=g_icon.xml&icon_url=https://oss.looodesign.com/test/img/%s.svg&mode=svg_path&demo=true&i=%s" width="300" height="300"></iframe>"""%(src_img_path , gy_img_path)
            else:
                rst += """<iframe src="https://oss.looodesign.com/test/img/%s.svg" width="300" height="300"></iframe>""" % (
                img_path.split('/')[-1][:-4])

        else:
            rst += """<iframe src="https://oss.looodesign.com/test/img/%s.svg" width="300" height="300"></iframe>"""%(img_path.split('/')[-1][:-4])
        return rst

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9001, debug=False)



