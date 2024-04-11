#!/usr/bin/env python3
"""
本文件启动一个同名节点替代PL-VINS中的feature_tracker_node
功能是监听图像信息并使用导入的自定义点特征提取模块来进行detecting&tracking
"""
import cv2

import copy
import torch

import numpy as np
from time import time

from LSD import MyLineExtractModel # 导入自定义线特征模型
from pylsd import lsd
import os

init_pub = False
count_frame = 0

if __name__ == '__main__':
    my_lineextract_model = MyLineExtractModel()  # 利用参数文件建立自定义线特征模型
    img = cv2.imread('250.png')
    keylines = my_lineextract_model.extract_line(img)
    out_img = my_lineextract_model.draw_line(img, keylines)
    cv2.imwrite('out.png', out_img)
    # full_name = '/home/plus/plvins_ws/src/PL-VINS/feature_tracker/scripts/LSD/250.png'
    # folder, img_name = os.path.split(full_name)
    # img = cv2.imread(full_name, cv2.IMREAD_COLOR)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # segments = lsd(img_gray, scale=0.5)

    # for i in range(segments.shape[0]):
    #     pt1 = (int(segments[i, 0]), int(segments[i, 1]))
    #     pt2 = (int(segments[i, 2]), int(segments[i, 3]))
    #     width = segments[i, 4]
    #     cv2.line(img, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))

    # cv2.imwrite(os.path.join(folder, 'cv2_' + img_name.split('.')[0] + '.jpg'), img)