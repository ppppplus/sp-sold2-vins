#!/usr/bin/env python3
"""
本文件定义了一个类来实现点特征提取的功能，替代PL-VINS源码中的feature_tracker.cpp
"""
import cv2
import copy
import numpy as np 
import rospy
from time import time

# from utils.PointTracker import PointTracker
# from utils.feature_process import SuperPointFrontend_torch, SuperPointFrontend
run_time = 0.0
match_time = 0.0

class FeatureTracker:
	def __init__(self, extract_model, match_model, camera_model, min_cnt=150, opencv=False):
		# point_model为自定义点特征模型类，其中提供extract方法接受一个图像输入，输出特征点信息
		self.extractor = extract_model
		self.matcher = match_model
		self.camera = camera_model
		self.new_frame = None
		self.allfeature_cnt = 0
		self.min_cnt = min_cnt
		self.no_display = True
		self.opencv = opencv
		if opencv:
			self.init_opencvmodel()
		else:
			self.init_model()

	def init_model(self):
		self.forwframe_ = {
				'PointID': [],
				'keyPoint': np.zeros((3,0)),
				'descriptor': np.zeros((256,0)),
				'image': None,
				}

		self.curframe_ = {
				'PointID': [],
				'keyPoint': np.zeros((3,0)),
				'descriptor': np.zeros((256,0)),
				'image': None
				}
	def init_opencvmodel(self):
		self.forwframe_ = {
				'PointID': [],
				'keyPoint': [],
				'descriptor': [],
				'image': None,
				}

		self.curframe_ = {
				'PointID': [],
				'keyPoint': [],
				'descriptor': [],
				'image': None
				}

	def undistortedPoints(self):

		cur_un_pts = copy.deepcopy(self.curframe_['keyPoint'])
		ids = copy.deepcopy(self.curframe_['PointID'])
		cur_pts = copy.deepcopy(self.curframe_['keyPoint'])
		un_img = copy.deepcopy(self.curframe_['image'])
		un_img = cv2.cvtColor(un_img, cv2.COLOR_GRAY2RGB)
		for i in range(cur_pts.shape[1]):
			b = self.camera.liftProjective(cur_pts[:2,i])
			cur_un_pts[0,i] = b[0] / b[2]	# x
			cur_un_pts[1,i] = b[1] / b[2]	# y
		# rospy.loginfo("get point x%f, y%f", cur_pts[0,i], cur_pts[1,i])
		return cur_un_pts, cur_pts, ids, un_img

	
	def readImage(self, new_img):
		self.new_frame = self.camera.undistortImg(new_img)
		self.first_image_flag = False

		if self.opencv:
			self.processcvImage()
		else:
			self.processImage()

		# assert(new_img.ndim==2 and new_img.shape[0]==self.height and new_img.shape[1]==self.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		
		
		# print("wsssssssssssssssssssssssssssssssssss:", self.new_frame.ndim)
		# cv2.imshow('new_frame', self.new_frame)
		# cv2.waitKey(0)
		
		
	def processImage(self):
		if not self.forwframe_['PointID']:
			self.forwframe_['PointID'] = []
			self.forwframe_['keyPoint'] = np.zeros((3,0))
			self.forwframe_['descriptor'] = np.zeros((256,0))

			self.forwframe_['image'] = self.new_frame
			self.curframe_['image'] = self.new_frame
			self.first_image_flag = True

		else:
			self.forwframe_['PointID'] = []
			self.forwframe_['keyPoint'] = np.zeros((3,0))
			self.forwframe_['descriptor'] = np.zeros((256,0))

			self.forwframe_['image'] = self.new_frame
		
		######################### 提取关键点和描述子 ############################
		print('*'*10 + " current frame " + '*'*10)
		start_time = time()
		self.forwframe_['keyPoint'], self.forwframe_['descriptor'] = self.extractor.extract(self.new_frame)
		pts_dim = self.forwframe_['keyPoint'].shape[0]
		desc_dim = self.forwframe_['descriptor'].shape[0]
		# print("pts_shape: {}, desc_shape: {}".format(self.forwframe_["keyPoint"].shape, self.forwframe_["descriptor"].shape))

		global run_time
		run_time += ( time()-start_time )
		print("point extraction time is:", time()-start_time)
		print("total run time is :", run_time)

		num_points = self.forwframe_['keyPoint'].shape[1]
		print("current keypoint size is :", num_points)

		# if keyPoint_size < self.min_cnt-50:
		# 	self.forwframe_['keyPoint'], self.forwframe_['descriptor'], heatmap = self.SuperPoint_Ghostnet.run(self.new_frame, conf_thresh=0.01)
		# 	keyPoint_size = self.forwframe_['keyPoint'].shape[1]
		# 	print("next keypoint size is ", keyPoint_size)

		for _ in range(num_points):
			if self.first_image_flag == True:
				self.forwframe_['PointID'].append(self.allfeature_cnt)
				self.allfeature_cnt = self.allfeature_cnt+1
			else:
				self.forwframe_['PointID'].append(-1)
		
		##################### 开始处理匹配的特征点 ###############################
		if self.curframe_['keyPoint'].shape[1] > 0:
			start_time = time()
			matches = self.matcher.match( 
									{
										"descriptors0": self.forwframe_['descriptor'],
	  									"descriptors1": self.curframe_['descriptor'],
										"keypoints0": self.forwframe_['keyPoint'],
										"keypoints1": self.curframe_["keyPoint"],
										"shape": self.forwframe_["image"].shape
									}
							)
			# matches: [3,num_matches]
			global match_time
			match_time += time()-start_time
			print("match time is :", match_time)
			print("match size is :", matches.shape[1])
			######################## 保证匹配得到的pointID相同 #####################
			for k in range(matches.shape[1]):
				self.forwframe_['PointID'][int(matches[0,k])] = self.curframe_['PointID'][int(matches[1,k])]

			################### 将跟踪的点与没跟踪的点进行区分 #####################
			vecpoint_new = np.zeros((pts_dim,0))
			vecpoint_tracked = np.zeros((pts_dim,0))
			pointID_new = []
			pointID_tracked = []
			descr_new = np.zeros((desc_dim,0))
			descr_tracked = np.zeros((desc_dim,0))

			for i in range(num_points):
				if self.forwframe_['PointID'][i] == -1 :
					self.forwframe_['PointID'][i] = self.allfeature_cnt
					self.allfeature_cnt = self.allfeature_cnt+1
					vecpoint_new = np.append(vecpoint_new, self.forwframe_['keyPoint'][:,i:i+1], axis=1)
					pointID_new.append(self.forwframe_['PointID'][i])
					descr_new = np.append(descr_new, self.forwframe_['descriptor'][:,i:i+1], axis=1)
				else:
					vecpoint_tracked = np.append(vecpoint_tracked, self.forwframe_['keyPoint'][:,i:i+1], axis=1)
					pointID_tracked.append(self.forwframe_['PointID'][i])
					descr_tracked = np.append(descr_tracked, self.forwframe_['descriptor'][:,i:i+1], axis=1)

			########### 跟踪的点特征少于阈值了，那就补充新的点特征 ###############

			diff_n = self.min_cnt - vecpoint_tracked.shape[1]
			if diff_n > 0:
				if vecpoint_new.shape[1] >= diff_n:
					for k in range(diff_n):
						vecpoint_tracked = np.append(vecpoint_tracked, vecpoint_new[:,k:k+1], axis=1)
						pointID_tracked.append(pointID_new[k])
						descr_tracked = np.append(descr_tracked, descr_new[:,k:k+1], axis=1)
				else:
					for k in range(vecpoint_new.shape[1]):
						vecpoint_tracked = np.append(vecpoint_tracked, vecpoint_new[:,k:k+1], axis=1)
						pointID_tracked.append(pointID_new[k])
						descr_tracked = np.append(descr_tracked, descr_new[:,k:k+1], axis=1)
			
			self.forwframe_['keyPoint'] = vecpoint_tracked
			self.forwframe_['PointID'] = pointID_tracked
			self.forwframe_['descriptor'] = descr_tracked

		self.curframe_ = copy.deepcopy(self.forwframe_)

	def processcvImage(self):
		if not self.forwframe_['PointID']:
			self.forwframe_['PointID'] = []
			self.forwframe_['keyPoint'] = []
			self.forwframe_['descriptor'] = []

			self.forwframe_['image'] = self.new_frame
			self.curframe_['image'] = self.new_frame
			self.first_image_flag = True

		else:
			self.forwframe_['PointID'] = []
			self.forwframe_['keyPoint'] = []
			self.forwframe_['descriptor'] = []

			self.forwframe_['image'] = self.new_frame
		
		######################### 提取关键点和描述子 ############################
		print('*'*10 + " current frame " + '*'*10)
		start_time = time()
		self.forwframe_['keyPoint'], self.forwframe_['descriptor'] = self.extractor.extract(self.new_frame)
		# pts_dim = self.forwframe_['keyPoint'].shape[0]
		# desc_dim = self.forwframe_['descriptor'].shape[0]
		# print("pts_shape: {}, desc_shape: {}".format(self.forwframe_["keyPoint"].shape, self.forwframe_["descriptor"].shape))

		global run_time
		run_time += ( time()-start_time )
		print("point extraction time is:", time()-start_time)
		print("total run time is :", run_time)

		num_points = len(self.forwframe_['keyPoint'])
		print("current keypoint size is :", num_points)

		# if keyPoint_size < self.min_cnt-50:
		# 	self.forwframe_['keyPoint'], self.forwframe_['descriptor'], heatmap = self.SuperPoint_Ghostnet.run(self.new_frame, conf_thresh=0.01)
		# 	keyPoint_size = self.forwframe_['keyPoint'].shape[1]
		# 	print("next keypoint size is ", keyPoint_size)

		for _ in range(num_points):
			if self.first_image_flag == True:
				self.forwframe_['PointID'].append(self.allfeature_cnt)
				self.allfeature_cnt = self.allfeature_cnt+1
			else:
				self.forwframe_['PointID'].append(-1)
		
		##################### 开始处理匹配的特征点 ###############################
		if len(self.curframe_['keyPoint']) > 0:
			start_time = time()
			matches = self.matcher.match( 
									{
										"descriptors0": self.forwframe_['descriptor'],
	  									"descriptors1": self.curframe_['descriptor']
									}
							)
			# matches: [3,num_matches]
			global match_time
			match_time += time()-start_time
			print("match time is :", match_time)
			print("match size is :", matches.shape[1])
			######################## 保证匹配得到的pointID相同 #####################
			for k in range(matches.shape[1]):
				self.forwframe_['PointID'][int(matches[0,k])] = self.curframe_['PointID'][int(matches[1,k])]

			################### 将跟踪的点与没跟踪的点进行区分 #####################
			vecpoint_new = []
			vecpoint_tracked = []
			pointID_new = []
			pointID_tracked = []
			descr_new = []
			descr_tracked = []

			for i in range(num_points):
				if self.forwframe_['PointID'][i] == -1 :
					self.forwframe_['PointID'][i] = self.allfeature_cnt
					self.allfeature_cnt = self.allfeature_cnt+1
					vecpoint_new.append(self.forwframe_['keyPoint'][i])
					pointID_new.append(self.forwframe_['PointID'][i])
					descr_new.append(self.forwframe_['descriptor'][i])
				else:
					vecpoint_tracked.append(self.forwframe_['keyPoint'][i])
					pointID_tracked.append(self.forwframe_['PointID'][i])
					descr_tracked.append(self.forwframe_['descriptor'][i])

			########### 跟踪的点特征少于阈值了，那就补充新的点特征 ###############

			diff_n = self.min_cnt - vecpoint_tracked.shape[1]
			if diff_n > 0:
				if len(vecpoint_new) >= diff_n:
					for k in range(diff_n):
						vecpoint_tracked.append(vecpoint_new[k])
						pointID_tracked.append(pointID_new[k])
						descr_tracked.append(descr_new[k])
				else:
					for k in range(len(vecpoint_new)):
						vecpoint_tracked.append(vecpoint_new[k])
						pointID_tracked.append(pointID_new[k])
						descr_tracked.append(descr_new[k])
			
			self.forwframe_['keyPoint'] = vecpoint_tracked
			self.forwframe_['PointID'] = pointID_tracked
			self.forwframe_['descriptor'] = descr_tracked

		self.curframe_['keyPoint'] = self.forwframe_['keyPoint']
		self.curframe_['PointID'] = self.curframe_['PointID']
		self.curframe_['descriptor'] = self.curframe_['descriptor']
		self.curframe_['image'] = self.curframe_['image']







# class FeatureTracker:
# 	def __init__(self, extract_model, match_model, camera_model, min_cnt=150):
# 		# point_model为自定义点特征模型类，其中提供extract方法接受一个图像输入，输出特征点信息
# 		self.extractor = extract_model
# 		self.matcher = match_model
# 		self.camera = camera_model
# 		self.forwframe_ = {
# 				'PointID': [],
# 				'keyPoint': np.zeros((3,0)),
# 				'descriptor': np.zeros((256,0)),
# 				'image': None,
# 				}

# 		self.curframe_ = {
# 				'PointID': [],
# 				'keyPoint': np.zeros((3,0)),
# 				'descriptor': np.zeros((256,0)),
# 				'image': None
# 				}
	
# 		self.new_frame = None
# 		self.allfeature_cnt = 0
# 		self.min_cnt = min_cnt
# 		self.no_display = True
		
# 		# self.cuda = opts.cuda
# 		# self.scale = opts.scale
# 		# 
# 		# self.nms_dist = opts.nms_dist
# 		# self.nn_thresh = opts.nn_thresh
# 		# self.no_display = opts.no_display
# 		# self.width = opts.W // opts.scale
# 		# self.height = opts.H // opts.scale
# 		# self.conf_thresh = opts.conf_thresh
# 		# self.weights_path = opts.weights_path

# 		# SuperPointFrontend_torch SuperPointFrontend
# 		# self.SuperPoint_Ghostnet = SuperPointFrontend_torch(
# 		# 	weights_path = self.weights_path, 
# 		# 	nms_dist = self.nms_dist,
# 		# 	conf_thresh = self.conf_thresh,
# 		# 	cuda = self.cuda
# 		# 	)
		
# 		# self.tracker = PointTracker(nn_thresh=self.nn_thresh)

# 	def undistortedPoints(self):

# 		cur_un_pts = copy.deepcopy(self.curframe_['keyPoint'])
# 		ids = copy.deepcopy(self.curframe_['PointID'])
# 		cur_pts = copy.deepcopy(self.curframe_['keyPoint'])
# 		un_img = copy.deepcopy(self.curframe_['image'])
# 		un_img = cv2.cvtColor(un_img, cv2.COLOR_GRAY2RGB)
# 		for i in range(cur_pts.shape[1]):
# 			b = self.camera.liftProjective(cur_pts[:2,i])
# 			cur_un_pts[0,i] = b[0] / b[2]	# x
# 			cur_un_pts[1,i] = b[1] / b[2]	# y
# 		# rospy.loginfo("get point x%f, y%f", cur_pts[0,i], cur_pts[1,i])
# 		return cur_un_pts, cur_pts, ids, un_img

	
# 	def readImage(self, new_img):

# 		# assert(new_img.ndim==2 and new_img.shape[0]==self.height and new_img.shape[1]==self.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		
# 		self.new_frame = self.camera.undistortImg(new_img)
# 		# print("wsssssssssssssssssssssssssssssssssss:", self.new_frame.ndim)
# 		# cv2.imshow('new_frame', self.new_frame)
# 		# cv2.waitKey(0)
		
# 		self.first_image_flag = False

# 		if not self.forwframe_['PointID']:
# 			self.forwframe_['PointID'] = []
# 			self.forwframe_['keyPoint'] = np.zeros((3,0))
# 			self.forwframe_['descriptor'] = np.zeros((256,0))

# 			self.forwframe_['image'] = self.new_frame
# 			self.curframe_['image'] = self.new_frame
# 			self.first_image_flag = True

# 		else:
# 			self.forwframe_['PointID'] = []
# 			self.forwframe_['keyPoint'] = np.zeros((3,0))
# 			self.forwframe_['descriptor'] = np.zeros((256,0))

# 			self.forwframe_['image'] = self.new_frame
		
# 		######################### 提取关键点和描述子 ############################
# 		print('*'*10 + " current frame " + '*'*10)
# 		start_time = time()
# 		self.forwframe_['keyPoint'], self.forwframe_['descriptor'] = self.extractor.extract(self.new_frame)
# 		pts_dim = self.forwframe_['keyPoint'].shape[0]
# 		desc_dim = self.forwframe_['descriptor'].shape[0]
# 		# print("pts_shape: {}, desc_shape: {}".format(self.forwframe_["keyPoint"].shape, self.forwframe_["descriptor"].shape))

# 		global run_time
# 		run_time += ( time()-start_time )
# 		print("point extraction time is:", time()-start_time)
# 		print("total run time is :", run_time)

# 		num_points = self.forwframe_['keyPoint'].shape[1]
# 		print("current keypoint size is :", num_points)

# 		# if keyPoint_size < self.min_cnt-50:
# 		# 	self.forwframe_['keyPoint'], self.forwframe_['descriptor'], heatmap = self.SuperPoint_Ghostnet.run(self.new_frame, conf_thresh=0.01)
# 		# 	keyPoint_size = self.forwframe_['keyPoint'].shape[1]
# 		# 	print("next keypoint size is ", keyPoint_size)

# 		for _ in range(num_points):
# 			if self.first_image_flag == True:
# 				self.forwframe_['PointID'].append(self.allfeature_cnt)
# 				self.allfeature_cnt = self.allfeature_cnt+1
# 			else:
# 				self.forwframe_['PointID'].append(-1)
		
# 		##################### 开始处理匹配的特征点 ###############################
# 		if self.curframe_['keyPoint'].shape[1] > 0:
# 			start_time = time()
# 			matches = self.matcher.match( 
# 									{
# 										"descriptors0": self.forwframe_['descriptor'],
# 	  									"descriptors1": self.curframe_['descriptor'],
# 										"keypoints0": self.forwframe_['keyPoint'],
# 										"keypoints1": self.curframe_["keyPoint"],
# 										"shape": self.forwframe_["image"].shape
# 									}
# 							)
# 			# matches: [3,num_matches]
# 			global match_time
# 			match_time += time()-start_time
# 			print("match time is :", match_time)
# 			print("match size is :", matches.shape[1])
# 			######################## 保证匹配得到的pointID相同 #####################
# 			for k in range(matches.shape[1]):
# 				self.forwframe_['PointID'][int(matches[0,k])] = self.curframe_['PointID'][int(matches[1,k])]

# 			################### 将跟踪的点与没跟踪的点进行区分 #####################
# 			vecpoint_new = np.zeros((pts_dim,0))
# 			vecpoint_tracked = np.zeros((pts_dim,0))
# 			pointID_new = []
# 			pointID_tracked = []
# 			descr_new = np.zeros((desc_dim,0))
# 			descr_tracked = np.zeros((desc_dim,0))

# 			for i in range(num_points):
# 				if self.forwframe_['PointID'][i] == -1 :
# 					self.forwframe_['PointID'][i] = self.allfeature_cnt
# 					self.allfeature_cnt = self.allfeature_cnt+1
# 					vecpoint_new = np.append(vecpoint_new, self.forwframe_['keyPoint'][:,i:i+1], axis=1)
# 					pointID_new.append(self.forwframe_['PointID'][i])
# 					descr_new = np.append(descr_new, self.forwframe_['descriptor'][:,i:i+1], axis=1)
# 				else:
# 					vecpoint_tracked = np.append(vecpoint_tracked, self.forwframe_['keyPoint'][:,i:i+1], axis=1)
# 					pointID_tracked.append(self.forwframe_['PointID'][i])
# 					descr_tracked = np.append(descr_tracked, self.forwframe_['descriptor'][:,i:i+1], axis=1)

# 			########### 跟踪的点特征少于阈值了，那就补充新的点特征 ###############

# 			diff_n = self.min_cnt - vecpoint_tracked.shape[1]
# 			if diff_n > 0:
# 				if vecpoint_new.shape[1] >= diff_n:
# 					for k in range(diff_n):
# 						vecpoint_tracked = np.append(vecpoint_tracked, vecpoint_new[:,k:k+1], axis=1)
# 						pointID_tracked.append(pointID_new[k])
# 						descr_tracked = np.append(descr_tracked, descr_new[:,k:k+1], axis=1)
# 				else:
# 					for k in range(vecpoint_new.shape[1]):
# 						vecpoint_tracked = np.append(vecpoint_tracked, vecpoint_new[:,k:k+1], axis=1)
# 						pointID_tracked.append(pointID_new[k])
# 						descr_tracked = np.append(descr_tracked, descr_new[:,k:k+1], axis=1)
			
# 			self.forwframe_['keyPoint'] = vecpoint_tracked
# 			self.forwframe_['PointID'] = pointID_tracked
# 			self.forwframe_['descriptor'] = descr_tracked

# 		self.curframe_ = copy.deepcopy(self.forwframe_)

