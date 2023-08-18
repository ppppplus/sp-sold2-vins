#!/usr/bin/env python3
"""
本文件启动一个同名节点替代PL-VINS中的feature_tracker_node
功能是监听图像信息并使用导入的自定义点特征提取模块来进行detecting&tracking
"""
import cv2
import os, sys
import copy
import rospy
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
from feature_tracker import FeatureTracker

from utils_point.my_point_model import create_pointextract_instance, create_pointmatch_instance

init_pub = False
count_frame = 0

def img_callback(img_msg, params_dict):
    feature_tracker = params_dict["feature_tracker"]
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
        feature_tracker.readImage(conver_img)

        if True :

            feature_points = PointCloud()
            id_of_point = ChannelFloat32()
            u_of_point = ChannelFloat32()
            v_of_point = ChannelFloat32()
            # velocity_x_of_point = ChannelFloat32()
            # velocity_y_of_point = ChannelFloat32()
            feature_points.header = img_msg.header
            feature_points.header.frame_id = "world"

            cur_un_pts, cur_pts, ids, cur_un_img = feature_tracker.undistortedPoints()

            for j in range(len(ids)):
                un_pts = Point32()
                un_pts.x = cur_un_pts[0,j]
                un_pts.y = cur_un_pts[1,j]
                un_pts.z = 1

                feature_points.points.append(un_pts)
                id_of_point.values.append(ids[j])
                u_of_point.values.append(cur_pts[0,j])
                v_of_point.values.append(cur_pts[1,j])
                # velocity_x_of_point.values.append(0.0)
                # velocity_y_of_point.values.append(0.0)

            feature_points.channels.append(id_of_point)
            feature_points.channels.append(u_of_point)
            feature_points.channels.append(v_of_point)
            # feature_points.channels.append(velocity_x_of_point)
            # feature_points.channels.append(velocity_y_of_point)

            pub_img.publish(feature_points)

            ptr_toImageMsg = Image()

            ptr_toImageMsg.header = img_msg.header
            ptr_toImageMsg.height = height
            ptr_toImageMsg.width = width
            ptr_toImageMsg.encoding = 'bgr8'

            # ptr_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
            ptr_image = cur_un_img

            for pt in cur_pts.T:
                pt2 = (int(round(pt[0])), int(round(pt[1])))
                cv2.circle(ptr_image, pt2, 2, (0, 255, 0), thickness=2)

            ptr_toImageMsg.data = np.array(ptr_image).tostring()
            pub_match.publish(ptr_toImageMsg)



if __name__ == '__main__':
    rospy.init_node('feature_tracker', anonymous=False)
    yamlPath = rospy.get_param("~config_path", "/home/plus/Work/plvins_ws/src/PL-VINS/config/feature_tracker/mtuav_v3_config.yaml")
    with open(yamlPath,'rb') as f:
      params = yaml.load(f, Loader=yaml.FullLoader)
      point_params = params["point_feature_cfg"]
      camera_params = params["camera_cfg"]

    my_pointextract_model = create_pointextract_instance(point_params)  # 建立自定义点特征提取模型
    my_pointmatch_model = create_pointmatch_instance(point_params)  # 建立自定义点特征匹配模型
    camera_model = CameraModel(camera_params)   
    CameraIntrinsicParam = camera_model.generateCameraModel()   # 建立相机模型
    feature_tracker = FeatureTracker(my_pointextract_model, my_pointmatch_model, CameraIntrinsicParam,
                                     min_cnt=point_params["min_cnt"]) # 利用点特征模型和相机模型生成点特征处理器
    
    image_topic = params["image_topic"]
    rospy.loginfo("Pointfeature Tracker initialization completed, waiting for img from topic: %s", image_topic)

    sub_img = rospy.Subscriber(image_topic, Image, img_callback, 
                               {"feature_tracker": feature_tracker, "H": point_params["H"], "W": point_params["W"]}, 
                               queue_size=100) # 监听图像，提取和追踪点特征并发布

    pub_img = rospy.Publisher("~feature", PointCloud, queue_size=1000)
    pub_match = rospy.Publisher("~feature_img", Image, queue_size=1000)

    rospy.spin()
