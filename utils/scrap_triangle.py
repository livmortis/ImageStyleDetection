# -*- coding: UTF-8 -*-
import requests
from requests import RequestException
from bs4 import BeautifulSoup
import re
from multiprocessing import Pool
import os
import time
# base_url='https://lkker-luo.oss-cn-beijing.aliyuncs.com/test/img/'
base_url='https://oss.looodesign.com/test/img/'



# def save_image(content,title,i):#创建对应分类的文件夹，并且保存图片。
#     dir_name_or=str(title)[:100].strip()
#     dir_name = re.sub("[\s+\.\!\/]+", "",dir_name_or)
#     print(dir_name)
#     dir_path='F:\spider\picture\zhainan2\{}'.format(dir_name)
#     try:
#         os.mkdir(dir_path)
#         print("创建文件夹成功")
#     except:
#         pass

#     file_path='{0}\{1}.{2}'.format(dir_path,str(i).zfill(3),'jpg')
#     if not os.path.exists(file_path):
#         with open(file_path,'wb') as f:
#             f.write(content)
#             f.close()
#         print("写入图片成功")



def save_svg(content, num):
    dir_path= "../data/connectivity/triangle_ori/svg/"
    try:
        os.mkdir(dir_path)
        print("创建文件夹成功")
    except:
        pass

    file_path='{0}/{1}.{2}'.format(dir_path,str(num),'svg')
    if not os.path.exists(file_path):
        with open(file_path,'wb') as f:
            f.write(content)
            f.close()
        print("写入图片成功")


def main(i):
    # url=base_url+str(i)
    url=base_url+str(i)+".svg"
    response = requests.get(url, timeout=5)
    # soup = BeautifulSoup(response.text, 'lxml')
    try:
        if response.status_code == 200:
            print("请求图片成功")
            # print(response.content)
            # save_image(response.content, title, i)
            save_svg(response.content, i)
        else:
            print("请求图片失败")
    except RequestException:
        print('请求图片出错',url)
        return None


if __name__ == '__main__':
    pool=Pool()
    pool.map(main,[i for i in range(318182,320000)])


































