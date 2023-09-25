#!/usr/bin/env python3
"""
本文件定义了一个类来实现点特征提取的功能，替代PL-VINS源码中的feature_tracker.cpp
"""
import cv2
import copy
import numpy as np 
import rospy
import torch
from time import time

# from utils.PointTracker import PointTracker
# from utils.feature_process import SuperPointFrontend_torch, SuperPointFrontend
run_time = 0.0
match_time = 0.0

class featuremapGenerator:
	def __init__(self, plextract_model, camera_model):
		# featuremapGenerator为一个特征图生成器，通过定义网络，接收图像输入输出推理得到的特征图
		self.extractor = plextract_model
		self.camera = camera_model
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		
		# self.cuda = opts.cuda
		# self.scale = opts.scale
		# 
		# self.nms_dist = opts.nms_dist
		# self.nn_thresh = opts.nn_thresh
		# self.no_display = opts.no_display
		# self.width = opts.W // opts.scale
		# self.height = opts.H // opts.scale
		# self.conf_thresh = opts.conf_thresh
		# self.weights_path = opts.weights_path

		# SuperPointFrontend_torch SuperPointFrontend
		# self.SuperPoint_Ghostnet = SuperPointFrontend_torch(
		# 	weights_path = self.weights_path, 
		# 	nms_dist = self.nms_dist,
		# 	conf_thresh = self.conf_thresh,
		# 	cuda = self.cuda
		# 	)
		
		# self.tracker = PointTracker(nn_thresh=self.nn_thresh)
	
	def processImage(self, new_img):

		# assert(new_img.ndim==2 and new_img.shape[0]==self.height and new_img.shape[1]==self.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		
		self.new_frame = self.camera.undistortImg(new_img)
		# print("wsssssssssssssssssssssssssssssssssss:", self.new_frame.ndim)
		# cv2.imshow('new_frame', self.new_frame)
		# cv2.waitKey(0)
				
		######################### 提取关键点线和描述子 ############################
		print('*'*10 + " current frame " + '*'*10)
		start_time = time()
		featuremap = self.extractor.extract(self.new_frame)
		# print("pts_shape: {}, desc_shape: {}".format(self.forwframe_["keyPoint"].shape, self.forwframe_["descriptor"].shape))
		end_time = time()
		global run_time
		run_time += ( end_time-start_time )
		print("point&line extraction time is:", end_time-start_time)
		print("total run time is :", run_time)
		heatmap = featuremap["heatmap"]
		junction = featuremap["junction"]
		coarse_desc = featuremap["coarse_desc"]
		return heatmap, junction, coarse_desc
		# self.curframe_ = copy.deepcopy(self.forwframe_)
