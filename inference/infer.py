from requests import RequestException
from bs4 import BeautifulSoup
import re
from multiprocessing import Pool
import os
import time
import getopt
import sys

import pandas as pd
import csv

import json

import requests
from flask import Flask, request, make_response
from utils.exception import StyleException, WrongParameterException
import traceback

cwd = os.getcwd()
upper_cwd_list = cwd.split('/')[1:-1]
upper_cwd = ''
for i in upper_cwd_list:
    i = '/' + str(i)
    upper_cwd = str(upper_cwd)+str(i)
sys.path.append(upper_cwd)

from flowjob import flow_model

app = Flask(__name__)

# base_url='https://lkker-luo.oss-cn-beijing.aliyuncs.com/test/img/'
# base_url = 'https://oss.looodesign.com/test/img/'
base_url = 'https://oss.looodesign.com/test/png-500/'


def inference(pngpath, gy, gyid):
    model = flow_model()
    output = model.inference_inet(pngpath, gy, gyid)
    return output



def png_2output(content, num, gy, gyid):
    png_path = "../../data/inferenceTempImg/png"

    try:
        os.mkdir(png_path)
    except:
        pass

    png_file_path = '{0}/{1}.{2}'.format(png_path, str(num), 'png')
    if not os.path.exists(png_file_path):
        with open(png_file_path, 'wb') as f:
            f.write(content)
            f.close()
    output = inference(png_path + '/' + str(num) + '.png', gy, gyid)
    os.system('rm %s' % (png_path + '/' + str(num) + '.png'))

    return output, num

def get_args(arg=None, required=True):
    if arg:
        v = request.args.get(arg)
        if (v is None) and required:
            return None
        else:
            return v
    else:
        return request.args
@app.route('/imgstyle', methods=['GET'])
def main():
    if request.method == 'GET':
        try:
            i = get_args('i')
            gy = get_args('gy')
            gyid = get_args('gyid')
            try:
                int(i)
            except ValueError as e:
                raise WrongParameterException('error type of i:'+str(i) )
            try:
                int(gyid)
            except ValueError as e:
                raise WrongParameterException('error type of gyid:'+str(gyid))
            if not isinstance(gy, str):
                raise WrongParameterException('error type of gy:' +str(gy))
            url = base_url + str(i) + ".png"
            print(i)
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    response_img = response.content
                else:
                    raise RequestException(str(i))
            except Exception as e:
                print(traceback.print_exc())
                print(str(i)+"请求图片失败")
                result = {}
                result['code'] = -2
                result['msg'] = 'download %s img error' % (e)
                result['data'] = None
                return json.dumps(result)


            output, num = png_2output(response_img, i, gy, gyid)
            result = {}
            result['code'] = 0
            result['msg'] = 'success'
            result['data'] = output
            return json.dumps(result, ensure_ascii=False)



        except Exception as e:
            print(traceback.print_exc())
            result = {}
            if isinstance(e, StyleException) :
                result['code'] = e.code
                result['msg'] = e.mesg
            else:
                result['code'] = -1
                result['msg'] = str(e)
            result['data'] = None
            return json.dumps(result)



if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    # pool = Pool()
    # pool.map(main, [i for i in range(100013,100015)])
    app.run(host='0.0.0.0', port=8002, debug=False)













