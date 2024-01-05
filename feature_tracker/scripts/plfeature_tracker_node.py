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
from plfeature_tracker import PLFeatureTracker

from utils_pl.my_pl_model import create_plextract_instance, create_pointmatch_instance, create_linematch_instance

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
            ###### points ######
            feature_points = PointCloud()
            id_of_point = ChannelFloat32()
            u_of_point = ChannelFloat32()
            v_of_point = ChannelFloat32()
            # velocity_x_of_point = ChannelFloat32()
            # velocity_y_of_point = ChannelFloat32()
            feature_points.header = img_msg.header
            feature_points.header.frame_id = "world"

            cur_un_pts, cur_pts, pts_ids, pts_cur_un_img = feature_tracker.undistortedPoints()

            ###### lines ######
            feature_lines = PointCloud()
            id_of_line = ChannelFloat32()
            u_of_endpoint = ChannelFloat32()
            v_of_endpoint = ChannelFloat32()    # u,v是线段端点
            # velocity_x_of_line = ChannelFloat32()
            # velocity_y_of_line = ChannelFloat32()
            feature_lines.header = img_msg.header
            feature_lines.header.frame_id = "world"

            cur_un_vecline, cur_vecline, line_ids, line_cur_un_img = feature_tracker.undistortedLineEndPoints()

            for j in range(len(pts_ids)):
                un_pts = Point32()
                un_pts.x = cur_un_pts[0,j]
                un_pts.y = cur_un_pts[1,j]
                un_pts.z = 1

                feature_points.points.append(un_pts)
                id_of_point.values.append(pts_ids[j])
                u_of_point.values.append(cur_pts[0,j])
                v_of_point.values.append(cur_pts[1,j])
                # velocity_x_of_point.values.append(0.0)
                # velocity_y_of_point.values.append(0.0)
            for k in range(len(line_ids)):
                unline_pts = Point32()
                unline_pts.x = cur_un_vecline[k,0,1]    # VINS后端的坐标系和opencv相同
                unline_pts.y = cur_un_vecline[k,0,0]
                unline_pts.z = 1

                # 向建立的点云消息中加入line信息
                feature_lines.points.append(unline_pts)
                id_of_line.values.append(line_ids[k])
                u_of_endpoint.values.append(cur_un_vecline[k,1,1])
                v_of_endpoint.values.append(cur_un_vecline[k,1,0])

            feature_points.channels.append(id_of_point)
            feature_points.channels.append(u_of_point)
            feature_points.channels.append(v_of_point)

            feature_lines.channels.append(id_of_line)
            feature_lines.channels.append(u_of_endpoint)
            feature_lines.channels.append(v_of_endpoint)
            # feature_points.channels.append(velocity_x_of_point)
            # feature_points.channels.append(velocity_y_of_point)

            pub_point_img.publish(feature_points)
            pub_line_img.publish(feature_lines)

            ###### publish imgs ######
            ptr_toImageMsg = Image()

            ptr_toImageMsg.header = img_msg.header
            ptr_toImageMsg.height = height
            ptr_toImageMsg.width = width
            ptr_toImageMsg.encoding = 'bgr8'

            # ptr_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
            ptr_image = pts_cur_un_img

            for pt in cur_pts.T:
                pt2 = (int(round(pt[0])), int(round(pt[1])))
                cv2.circle(ptr_image, pt2, 2, (0, 255, 0), thickness=2)

            ptr_toImageMsg.data = np.array(ptr_image).tostring()
            pub_point_match.publish(ptr_toImageMsg)
            ###############################################################
            lineptr_toImageMsg = Image()

            lineptr_toImageMsg.header = img_msg.header
            lineptr_toImageMsg.height = height 
            lineptr_toImageMsg.width = width
            lineptr_toImageMsg.encoding = 'bgr8'

            # ptr_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
            lineptr_image = line_cur_un_img

            for j in range(len(line_ids)):
                pt1 = (int(round(cur_vecline[j,0,1])), int(round(cur_vecline[j,0,0])))
                pt2 = (int(round(cur_vecline[j,1,1])), int(round(cur_vecline[j,1,0])))
                # cv2.circle(ptr_image, pt2, 2, (0, 255, 0), thickness=2)
                cv2.line(lineptr_image, pt1, pt2, (0, 0, 255), 2)

            lineptr_toImageMsg.data = np.array(lineptr_image).tostring()
            pub_line_match.publish(lineptr_toImageMsg)



if __name__ == '__main__':
    rospy.init_node('feature_tracker', anonymous=False)
    yamlPath = rospy.get_param("~config_path", "/home/nnplvio_ws/src/sp-sold2-vins/config/feature_tracker/sp-sold2_config.yaml")
    with open(yamlPath,'rb') as f:
      params = yaml.load(f, Loader=yaml.FullLoader)
      pl_params = params["pl_feature_cfg"]
      point_params = params["point_feature_cfg"]
      line_params = params["line_feature_cfg"]

      camera_params = params["camera_cfg"]

    my_plextract_model = create_plextract_instance(pl_params)
    my_pointmatch_model = create_pointmatch_instance(point_params)  # 建立自定义点特征匹配模型
    my_linematch_model = create_linematch_instance(line_params)  # 建立自定义点特征匹配模型
    
    camera_model = CameraModel(camera_params)   
    CameraIntrinsicParam = camera_model.generateCameraModel()   # 建立相机模型
    feature_tracker = PLFeatureTracker(my_plextract_model, my_pointmatch_model, my_linematch_model, CameraIntrinsicParam,
                                     num_samples=line_params["num_samples"], min_point_cnt=point_params["min_cnt"], min_line_cnt=line_params["min_cnt"]) # 利用点特征模型和相机模型生成点特征处理器
    
    image_topic = params["image_topic"]
    rospy.loginfo("PLfeature Tracker initialization completed, waiting for img from topic: %s", image_topic)

    sub_img = rospy.Subscriber(image_topic, Image, img_callback, 
                               {"feature_tracker": feature_tracker, "H": pl_params["H"], "W": pl_params["W"]}, 
                               queue_size=100) # 监听图像，提取和追踪点线特征并发布

    pub_point_img = rospy.Publisher("/feature_tracker/feature", PointCloud, queue_size=1000)
    pub_point_match = rospy.Publisher("/feature_tracker/feature_img", Image, queue_size=1000)
    pub_line_img = rospy.Publisher("/linefeature_tracker/linefeature", PointCloud, queue_size=1000)
    pub_line_match = rospy.Publisher("/linefeature_tracker/linefeature_img", Image, queue_size=1000)

    rospy.spin()
