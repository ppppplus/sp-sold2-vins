#!/usr/bin/env python3
"""
本文件启动一个同名节点替代PL-VINS中的feature_tracker_node
功能是监听图像信息并使用导入的自定义点特征提取模块来进行detecting&tracking
"""
import cv2

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
from linefeature_tracker import LineFeatureTracker

from utils_line.my_line_model import create_lineextract_instance, create_linematch_instance

init_pub = False
count_frame = 0
opencv_method = ["tplsd"]

def img_callback(img_msg, params_dict):
    # 处理传入的图像，提取线特征
    # 输入：图像msg和线特征提取器
    # 输出：无返回值，发布线特征PointCloud
    global init_pub
    global count_frame
    linefeature_tracker = params_dict["feature_tracker"]
    height = params_dict["H"]
    width = params_dict["W"]
    if not init_pub :
        init_pub = True
    else :
        init_pub = False

        bridge = CvBridge()
        conver_img = bridge.imgmsg_to_cv2(img_msg, "mono8")

        # cur_img, status = read_image(conver_img, [param.height, param.width])

        # if status is False:
        #     print("Load image error, Please check image_info topic")
        #     return
        
        # scale = 2
        linefeature_tracker.readImage(conver_img)

        if True :

            feature_lines = PointCloud()
            id_of_line = ChannelFloat32()
            u_of_endpoint = ChannelFloat32()
            v_of_endpoint = ChannelFloat32()    # u,v是线段端点
            # velocity_x_of_line = ChannelFloat32()
            # velocity_y_of_line = ChannelFloat32()
            feature_lines.header = img_msg.header
            feature_lines.header.frame_id = "world"

            cur_un_vecline, cur_vecline, ids, cur_un_img = linefeature_tracker.undistortedLineEndPoints()

            for j in range(len(ids)):
                un_pts = Point32()
                un_pts.x = cur_un_vecline[j,0,1]    # VINS后端的坐标系和opencv相同
                un_pts.y = cur_un_vecline[j,0,0]
                un_pts.z = 1

                # 向建立的点云消息中加入line信息
                feature_lines.points.append(un_pts)
                id_of_line.values.append(ids[j])
                u_of_endpoint.values.append(cur_un_vecline[j,1,1])
                v_of_endpoint.values.append(cur_un_vecline[j,1,0])
                # velocity_x_of_line.values.append(0.0)
                # velocity_y_of_line.values.append(0.0)

            feature_lines.channels.append(id_of_line)
            feature_lines.channels.append(u_of_endpoint)
            feature_lines.channels.append(v_of_endpoint)
            # feature_lines.channels.append(velocity_x_of_line)
            # feature_lines.channels.append(velocity_y_of_line)

            pub_img.publish(feature_lines)

            ptr_toImageMsg = Image()

            ptr_toImageMsg.header = img_msg.header
            ptr_toImageMsg.height = height 
            ptr_toImageMsg.width = width
            ptr_toImageMsg.encoding = 'bgr8'

            # ptr_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")
            ptr_image = cur_un_img

            for j in range(len(ids)):
                pt1 = (int(round(cur_vecline[j,0,1])), int(round(cur_vecline[j,0,0])))
                pt2 = (int(round(cur_vecline[j,1,1])), int(round(cur_vecline[j,1,0])))
                # cv2.circle(ptr_image, pt2, 2, (0, 255, 0), thickness=2)
                cv2.line(ptr_image, pt1, pt2, (0, 0, 255), 2)

            ptr_toImageMsg.data = np.array(ptr_image).tostring()
            pub_match.publish(ptr_toImageMsg)



if __name__ == '__main__':

    rospy.init_node('linefeature_tracker', anonymous=False)
    
    # yamlPath = '/home/nvidia/plvins_ws/src/PL-VINS/feature_tracker/config/config.yaml'
    yamlPath = rospy.get_param("~config_path", "/home/nnplvio_ws/src/sp-sold2-vins/config/feature_tracker/test_config.yaml")
    # print(yamlPath)
    with open(yamlPath,'rb') as f:
      # yaml文件通过---分节，多个节组合成一个列表
      params = yaml.load(f, Loader=yaml.FullLoader)
      line_params = params["line_feature_cfg"]
      camera_params = params["camera_cfg"]
      extract_method = line_params["extract_method"]
      opencv = 1 if extract_method in opencv_method else 0
      
    my_line_extract_model = create_lineextract_instance(line_params)  # 利用参数文件建立自定义线特征模型
    my_line_match_model = create_linematch_instance(line_params)

    # CamearIntrinsicParam = PinholeCamera(
    #     fx = 461.6, fy = 460.3, cx = 363.0, cy = 248.1, 
    #     k1 = -2.917e-01, k2 = 8.228e-02, p1 = 5.333e-05, p2 = -1.578e-04
    #     )  

    camera_model = CameraModel(camera_params)
    CameraIntrinsicParam = camera_model.generateCameraModel()
    linefeature_tracker = LineFeatureTracker(my_line_extract_model, my_line_match_model, CameraIntrinsicParam, 
                                             num_samples=line_params["num_samples"], min_cnt=line_params["min_cnt"], opencv=opencv) # 利用点特征模型和相机模型生成点特征处理器

    image_topic = params["image_topic"]
    rospy.loginfo("Linefeature Tracker initialization completed, waiting for img from topic: %s", image_topic)
    sub_img = rospy.Subscriber(image_topic, Image, img_callback, 
                               {"feature_tracker": linefeature_tracker, "H": line_params["H"], "W": line_params["W"]}, 
                               queue_size=100)


    pub_img = rospy.Publisher("/linefeature_tracker/linefeature", PointCloud, queue_size=1000)
    pub_match = rospy.Publisher("/linefeature_tracker/linefeature_img", Image, queue_size=1000)

    rospy.spin()
