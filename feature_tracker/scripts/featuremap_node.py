#!/usr/bin/env python3
import cv2
import os, sys
import copy

import rospy
import torch
import yaml

import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from sensor_msgs.msg import Image
from time import time

from utils.parameter import read_image
from utils.camera_model import CameraModel
from utils_pl.featuremap_model import SPSOLD2Model
from featuremap import featuremapGenerator
from feature_tracker.msg import Featuremap

from utils_pl.my_pl_model import create_plextract_instance, create_pointmatch_instance, create_linematch_instance

init_pub = False
count_frame = 0


def img_callback(img_msg, params_dict):
    featuremap_generator = params_dict["featuremap_generator"]
    height = params_dict["H"]
    width = params_dict["W"]
    global init_pub
    global count_frame

    if not init_pub :
        init_pub = True
    else :
        init_pub = False

        bridge = CvBridge()
        conver_img = bridge.imgmsg_to_cv2(img_msg, "mono8")
        
        # scale = 2
        heatmap, junction, coarse_desc = featuremap_generator.processImage(conver_img)
        # heatmap = featuremap["heatmap"]
        # junction = featuremap["junction"]
        # coarse_desc = featuremap["coarse_desc"]
        heatmap_msg = Float32MultiArray()
        junction_msg = Float32MultiArray()
        coarse_desc_msg = Float32MultiArray()
        featuremap_msg = Featuremap()
        print(heatmap.shape, junction.shape, coarse_desc.shape)
        heatmap_c, heatmap_h, heatmap_w = heatmap.shape
        junction_c, junction_h, junction_w = junction.shape
        coarse_desc_c, coarse_desc_h, coarse_desc_w = coarse_desc.shape
        
        heatmap_c_dim = MultiArrayDimension(label="channel", size=heatmap_c, stride=heatmap_c)
        heatmap_w_dim = MultiArrayDimension(label="width",  size=heatmap_w, stride=heatmap_w*heatmap_c)
        heatmap_h_dim = MultiArrayDimension(label="height", size=heatmap_h, stride=heatmap_h*heatmap_w*heatmap_c)
        
        junction_c_dim = MultiArrayDimension(label="channel", size=junction_c, stride=junction_c)
        junction_w_dim = MultiArrayDimension(label="width",  size=junction_w, stride=junction_w*junction_c)
        junction_h_dim = MultiArrayDimension(label="height", size=junction_h, stride=junction_h*junction_w*junction_c)
        
        coarse_desc_c_dim = MultiArrayDimension(label="channel", size=coarse_desc_c, stride=coarse_desc_c)
        coarse_desc_w_dim = MultiArrayDimension(label="width",  size=coarse_desc_w, stride=coarse_desc_w*coarse_desc_c)
        coarse_desc_h_dim = MultiArrayDimension(label="height", size=coarse_desc_h, stride=coarse_desc_h*coarse_desc_w*coarse_desc_c)
        
        heatmap_msg.layout = MultiArrayLayout(dim=[heatmap_h_dim, heatmap_w_dim, heatmap_c_dim], data_offset=0)
        heatmap_msg.data = heatmap.reshape(-1).tolist()
        # print("heatmap_msg-layout: ", [heatmap_msg.layout.dim[0].size,
        #                                heatmap_msg.layout.dim[1].size,
        #                                heatmap_msg.layout.dim[2].size])
        
        junction_msg.layout = MultiArrayLayout(dim=[junction_h_dim, junction_w_dim, junction_c_dim], data_offset=0)
        junction_msg.data = junction.reshape(-1).tolist()
        
        coarse_desc_msg.layout = MultiArrayLayout(dim=[coarse_desc_h_dim, coarse_desc_w_dim, coarse_desc_c_dim], data_offset=0)
        coarse_desc_msg.data = coarse_desc.reshape(-1).tolist()
        
        featuremap_msg.heatmap = heatmap_msg
        featuremap_msg.junction = junction_msg
        featuremap_msg.coarse_desc = coarse_desc_msg
        
        pub_featuremap.publish(featuremap_msg)



if __name__ == '__main__':
    rospy.init_node('featuremap_generator', anonymous=False)
    yamlPath = rospy.get_param("~config_path", "/home/nvidia/Work/sp-sold2-vins_ws/src/sp-sold2-vins/config/feature_tracker/sp-sold2net_config.yaml")
    with open(yamlPath,'rb') as f:
      params = yaml.load(f, Loader=yaml.FullLoader)
      model_params = params["model_cfg"]
    #   pl_params = params["pl_feature_cfg"]
    #   point_params = params["point_feature_cfg"]
    #   line_params = params["line_feature_cfg"]
    
      camera_params = params["camera_cfg"]

    my_model = SPSOLD2Model(model_params)
    # my_pointmatch_model = create_pointmatch_instance(point_params)  # 建立自定义点特征匹配模型
    # my_linematch_model = create_linematch_instance(line_params)  # 建立自定义点特征匹配模型
    
    
    camera_model = CameraModel(camera_params)   
    CameraIntrinsicParam = camera_model.generateCameraModel()   # 建立相机模型
    # feature_tracker = PLFeatureTracker(my_plextract_model, my_pointmatch_model, my_linematch_model, CameraIntrinsicParam,
    #                                  num_samples=line_params["num_samples"], min_point_cnt=point_params["min_cnt"], min_line_cnt=line_params["min_cnt"]) # 利用点特征模型和相机模型生成点特征处理器
    featuremap_generator = featuremapGenerator(my_model, CameraIntrinsicParam) # 利用点线特征模型和相机模型得到特征图生成器
    
    image_topic = params["image_topic"]
    rospy.loginfo("PLfeature Tracker initialization completed, waiting for img from topic: %s", image_topic)

    sub_img = rospy.Subscriber(image_topic, Image, img_callback, 
                               {"featuremap_generator": featuremap_generator, "H": model_params["H"], "W": model_params["W"]}, 
                               queue_size=100) # 监听图像，提取和追踪点线特征并发布

    # pub_point_img = rospy.Publisher("~feature", PointCloud, queue_size=1000).
    # pub_point_match = rospy.Publisher("~feature_img", Image, queue_size=1000)
    # pub_line_img = rospy.Publisher("~linefeature", PointCloud, queue_size=1000)
    # pub_line_match = rospy.Publisher("~linefeature_img", Image, queue_size=1000)
    pub_featuremap = rospy.Publisher("/feature_map", Featuremap, queue_size=100)

    rospy.spin()
