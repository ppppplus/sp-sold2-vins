#!/usr/bin/env python3
"""
本文件启动一个同名节点替代PL-VINS中的feature_tracker_node
功能是监听图像信息并使用导入的自定义点特征提取模块来进行detecting&tracking
"""
import cv2
import os, sys
import copy
# import rospy
import torch
import yaml

import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32
from time import time

from utils.parameter import read_image
from utils.camera_model import CameraModel
from pointfeature_tracker import FeatureTracker

from utils_point.my_point_model import create_pointextract_instance, create_pointmatch_instance


if __name__ == '__main__':
    # yamlPath = "/home/nnplvio_ws/src/sp-sold2-vins/config/feature_tracker/euroc_config.yaml"
    params = {
        "point_feature_cfg":{
           "H": 480, # Input image height
            "W": 752, # Input image width
            "min_cnt": 150,
            "extract_method": "r2d2",
            "r2d2":{
                "weights_path": "/home/nnplvio_ws/src/sp-sold2-vins/feature_tracker/scripts/utils_point/r2d2/r2d2_WASF_N16.pt",
                "reliability_thr": 0.7,
                "repeatability_thr": 0.7,
                "cuda": True
            },
            
            "match_method": "knn",
            "knn": {
                "thresh": 0.85
            }


        }
    }

    point_params = params["point_feature_cfg"]
    # camera_params = params["camera_cfg"]

    my_pointextract_model = create_pointextract_instance(point_params)  # 建立自定义点特征提取模型
    my_pointmatch_model = create_pointmatch_instance(point_params)
    img0 = cv2.imread("/home/nnplvio_ws/src/sp-sold2-vins/feature_tracker/scripts/img/img0.JPG")
    img1 = cv2.imread("/home/nnplvio_ws/src/sp-sold2-vins/feature_tracker/scripts/img/img1.JPG")
 
    pts0, desc0 = my_pointextract_model.extract(img0)
    pts1, desc1 = my_pointextract_model.extract(img1)
    print(desc1.shape, desc0.shape)
    print(pts0)
    
    # pts = np.array(pts)
    # desc = np.array(desc)
    # print(pts.shape, desc.shape)
    for i in range(pts0.shape[1]):
        pt2 = (int(round(pts0[0, i])), int(round(pts0[1, i])))
        cv2.circle(img0, pt2, 2, (0, 255, 0), thickness=2)
    # outimg1 = cv2.drawKeypoints(img0, keypoints=pts, outImage=None)
    
    cv2.imshow("Key Points", img0)
    cv2.waitKey(0)
    # data = {"descriptors0": desc0, "descriptors1":desc1}
    # match = my_pointmatch_model.match(data)
    # print(match.shape)
    # print(len(match), match[1].queryIdx, match[1].trainIdx)
    # outimage = cv2.drawMatches(img0, kp0, img1, kp1, match, outImg=None)
    # cv2.imshow("Match Result", outimage)
    # cv2.waitKey(0)
    # my_pointmatch_model = create_pointmatch_instance(point_params)  # 建立自定义点特征匹配模型
    # camera_model = CameraModel(camera_params)   
    # CameraIntrinsicParam = camera_model.generateCameraModel()   # 建立相机模型
    # feature_tracker = FeatureTracker(my_pointextract_model, my_pointmatch_model, CameraIntrinsicParam,
    #                                  min_cnt=point_params["min_cnt"]) # 利用点特征模型和相机模型生成点特征处理器
    
    # image_topic = params["image_topic"]
    # rospy.loginfo("Pointfeature Tracker initialization completed, waiting for img from topic: %s", image_topic)

    # sub_img = rospy.Subscriber(image_topic, Image, img_callback, 
    #                            {"feature_tracker": feature_tracker, "H": point_params["H"], "W": point_params["W"]}, 
    #                            queue_size=100) # 监听图像，提取和追踪点特征并发布

    # pub_img = rospy.Publisher("/feature_tracker/feature", PointCloud, queue_size=1000)
    # pub_match = rospy.Publisher("/feature_tracker/feature_img", Image, queue_size=1000)

    # rospy.spin()
