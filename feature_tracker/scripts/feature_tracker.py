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

myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])


class PLFeatureTracker:
	def __init__(self, plextract_model, pointmatch_model, linematch_model, camera_model, num_samples=8, min_point_cnt=150, min_line_cnt=50):
		# point_model为自定义点特征模型类，其中提供extract方法接受一个图像输入，输出特征点信息
		self.extractor = plextract_model
		self.point_matcher = pointmatch_model
		self.line_matcher = linematch_model
		self.camera = camera_model
		self.num_samples = num_samples
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.forwframe_ = {
				'PointID': None,
				'keyPoint': np.zeros((3,0)),
				'vecline': np.zeros((0,2,2)),
				'lineID': None,
				'pointdescriptor': torch.zeros((128,0)).to(self.device),
				'linedescriptor': torch.zeros((128,0)).to(self.device),
				'valid_points': None,	# 存储哪些线采样点为有效的
				'image': None,
				}

		self.curframe_ = {
				'PointID': None,
				'keyPoint': np.zeros((3,0)),
				'vecline': np.zeros((0,2,2)),
				'lineID': None,
				'pointdescriptor': torch.zeros((128,0)).to(self.device),
				'linedescriptor': torch.zeros((128,0)).to(self.device),
				'valid_points': None,	# 存储哪些线采样点为有效的
				'image': None,
				}
	
		self.new_frame = None
		self.all_pointfeature_cnt = 0
		self.all_linefeature_cnt = 0
		self.min_point_cnt = min_point_cnt
		self.min_line_cnt = min_line_cnt
		self.no_display = True
		
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
	
	def undistortedLineEndPoints(self):

		cur_un_vecline = copy.deepcopy(self.curframe_['vecline'])
		cur_vecline = copy.deepcopy(self.curframe_['vecline'])
		ids = copy.deepcopy(self.curframe_['lineID'])
		un_img = copy.deepcopy(self.curframe_['image'])
		un_img = cv2.cvtColor(un_img, cv2.COLOR_GRAY2RGB)
		

		for i in range(cur_vecline.shape[0]):
			b0 = self.camera.liftProjective(cur_vecline[i,0,:])
			b1 = self.camera.liftProjective(cur_vecline[i,1,:])
			cur_un_vecline[i,0,0] = b0[0] / b0[2]
			cur_un_vecline[i,0,1] = b0[1] / b0[2]
			cur_un_vecline[i,1,0] = b1[0] / b1[2]
			cur_un_vecline[i,1,1] = b1[1] / b1[2]
			# rospy.loginfo("get line sx%f, sy%f, ex%f, ey%f", cur_vecline[i,0,0], cur_vecline[i,0,1], cur_vecline[i,1,0], cur_vecline[i,1,1])
		return cur_un_vecline, cur_vecline, ids, un_img

	
	def readImage(self, new_img):

		# assert(new_img.ndim==2 and new_img.shape[0]==self.height and new_img.shape[1]==self.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		
		self.new_frame = self.camera.undistortImg(new_img)
		# print("wsssssssssssssssssssssssssssssssssss:", self.new_frame.ndim)
		# cv2.imshow('new_frame', self.new_frame)
		# cv2.waitKey(0)
		
		first_image_flag = False

		if not self.forwframe_['PointID']:
			self.forwframe_['PointID'] = []
			self.forwframe_['lineID'] = []

			self.forwframe_['image'] = self.new_frame
			self.curframe_['image'] = self.new_frame
			first_image_flag = True

		else:
			self.forwframe_['PointID'] = []
			self.forwframe_['lineID'] = []

			self.forwframe_['image'] = self.new_frame
		
		######################### 提取关键点线和描述子 ############################
		print('*'*10 + " current frame " + '*'*10)
		start_time = time()
		self.forwframe_['keyPoint'], self.forwframe_['pointdescriptor'], self.forwframe_['vecline'], self.forwframe_['linedescriptor'], self.forwframe_['valid_points'] = self.extractor.extract(self.new_frame)
		# print("pts_shape: {}, desc_shape: {}".format(self.forwframe_["keyPoint"].shape, self.forwframe_["descriptor"].shape))
		end_time = time()
		global run_time
		run_time += ( end_time-start_time )
		print("point&line extraction time is:", end_time-start_time)
		print("total run time is :", run_time)

		num_points = self.forwframe_['keyPoint'].shape[1]
		num_lines = self.forwframe_['vecline'].shape[0]
		print("current keypoint size is :", num_points)
		print("current number of lines is :", num_lines)

		# if keyPoint_size < self.min_cnt-50:
		# 	self.forwframe_['keyPoint'], self.forwframe_['descriptor'], heatmap = self.SuperPoint_Ghostnet.run(self.new_frame, conf_thresh=0.01)
		# 	keyPoint_size = self.forwframe_['keyPoint'].shape[1]
		# 	print("next keypoint size is ", keyPoint_size)

		for _ in range(num_points):
			if first_image_flag == True:
				self.forwframe_['PointID'].append(self.all_pointfeature_cnt)
				self.all_pointfeature_cnt = self.all_pointfeature_cnt+1
			else:
				self.forwframe_['PointID'].append(-1)
		for _ in range(num_lines):
			if first_image_flag == True:
				self.forwframe_['lineID'].append(self.all_linefeature_cnt)
				self.all_linefeature_cnt = self.all_linefeature_cnt+1
			else:
				self.forwframe_['lineID'].append(-1)
		
		##################### 开始处理匹配的特征点 ###############################
		if self.curframe_['keyPoint'].shape[1] > 0:
			start_time = time()
			point_matches = self.point_matcher.match( 
									{
										"descriptors0": self.forwframe_['pointdescriptor'],
	  									"descriptors1": self.curframe_['pointdescriptor'],
										"keypoints0": self.forwframe_['keyPoint'],
										"keypoints1": self.curframe_["keyPoint"],
										"shape": self.forwframe_["image"].shape
									}
							)
			# matches: [3,num_matches]
			print("pointmatch time is :", time()-start_time)
			print("pointmatch size is :", point_matches.shape[1])
			######################## 保证匹配得到的pointID相同 #####################
			for k in range(point_matches.shape[1]):
				# print(int(point_matches[0,k]), int(point_matches[1,k]), len(self.forwframe_['PointID']),  len(self.curframe_['PointID']))
				self.forwframe_['PointID'][int(point_matches[0,k])] = self.curframe_['PointID'][int(point_matches[1,k])]
				
			################### 将跟踪的点与没跟踪的点进行区分 #####################
			vecpoint_new = np.zeros((3,0))
			vecpoint_tracked = np.zeros((3,0))
			pointID_new = []
			pointID_tracked = []
			pointdescr_new = torch.zeros((128,0)).to(self.device)
			pointdescr_tracked = torch.zeros((128,0)).to(self.device)

			for i in range(num_points):
				if self.forwframe_['PointID'][i] == -1 :
					self.forwframe_['PointID'][i] = self.all_pointfeature_cnt
					self.all_pointfeature_cnt = self.all_pointfeature_cnt+1
					vecpoint_new = np.append(vecpoint_new, self.forwframe_['keyPoint'][:,i:i+1], axis=1)
					pointID_new.append(self.forwframe_['PointID'][i])
					pointdescr_new = torch.cat((pointdescr_new, self.forwframe_['pointdescriptor'][:,i:i+1]), dim=1)
				else:
					vecpoint_tracked = np.append(vecpoint_tracked, self.forwframe_['keyPoint'][:,i:i+1], axis=1)
					pointID_tracked.append(self.forwframe_['PointID'][i])
					pointdescr_tracked = torch.cat((pointdescr_tracked, self.forwframe_['pointdescriptor'][:,i:i+1]), dim=1)

			########### 跟踪的点特征少于阈值了，那就补充新的点特征 ###############

			diff_n = self.min_point_cnt - vecpoint_tracked.shape[1]
			if diff_n > 0:
				if vecpoint_new.shape[1] >= diff_n:
					for k in range(diff_n):
						vecpoint_tracked = np.append(vecpoint_tracked, vecpoint_new[:,k:k+1], axis=1)
						pointID_tracked.append(pointID_new[k])
						pointdescr_tracked = torch.cat((pointdescr_tracked, pointdescr_new[:,k:k+1]), dim=1)
				else:
					for k in range(vecpoint_new.shape[1]):
						vecpoint_tracked = np.append(vecpoint_tracked, vecpoint_new[:,k:k+1], axis=1)
						pointID_tracked.append(pointID_new[k])
						pointdescr_tracked = torch.cat((pointdescr_tracked, pointdescr_new[:,k:k+1]), dim=1)
			
			self.forwframe_['keyPoint'] = vecpoint_tracked
			self.forwframe_['PointID'] = pointID_tracked
			self.forwframe_['pointdescriptor'] = pointdescr_tracked

			##################### 开始处理匹配的线特征 ###############################
			if self.curframe_['vecline'].shape[0] > 0:
				start_time = time()
				index_lines1, index_lines2 = self.line_matcher.match( 
										self.forwframe_['vecline'],
										self.curframe_['vecline'],
										self.forwframe_['linedescriptor'][None,...], 
										self.curframe_['linedescriptor'][None,...],
										self.forwframe_['valid_points'],
										self.curframe_['valid_points']
								)
				# print("index:", index_lines1, index_lines2)
				print("line match time is :", time()-start_time)
				print("line match size is :", index_lines1.shape[0])
				######################## 保证匹配得到的lineID相同 #####################
				for k in range(index_lines1.shape[0]):
					self.forwframe_['lineID'][index_lines1[k]] = self.curframe_['lineID'][index_lines2[k]]

				################### 将跟踪的线与没跟踪的线进行区分 #####################
				vecline_new = np.zeros((0,2,2))
				vecline_tracked = np.zeros((0,2,2))
				validpoints_new = np.zeros((0,self.num_samples)).astype(int)
				validpoints_tracked = np.zeros((0,self.num_samples)).astype(int)
				lineID_new = []
				lineID_tracked = []
				linedescr_new = torch.zeros((128,0,self.num_samples)).to(self.device)
				linedescr_tracked = torch.zeros((128,0,self.num_samples)).to(self.device)

				for i in range(num_lines):
					if self.forwframe_['lineID'][i] == -1 :	# -1表示当前ID对应的line没有track到
						self.forwframe_['lineID'][i] = self.all_linefeature_cnt	# 没有跟踪到的线则编号为新的
						self.all_linefeature_cnt = self.all_linefeature_cnt+1
						vecline_new = np.append(vecline_new, self.forwframe_['vecline'][i:i+1,...], axis=0)	# 取出没有跟踪到的线信息并放入下一帧
						lineID_new.append(self.forwframe_['lineID'][i])
						linedescr_new = torch.cat((linedescr_new, self.forwframe_['linedescriptor'][:,i:i+1,:]), dim=1)
						validpoints_new = np.append(validpoints_new, self.forwframe_['valid_points'][i:i+1,:], axis=0)
					else:
						# 当前line已被track
						lineID_tracked.append(self.forwframe_['lineID'][i])
						vecline_tracked = np.append(vecline_tracked, self.forwframe_['vecline'][i:i+1,...], axis=0)
						linedescr_tracked = torch.cat((linedescr_tracked, self.forwframe_['linedescriptor'][:,i:i+1,:]), dim=1)
						validpoints_tracked = np.append(validpoints_tracked, self.forwframe_['valid_points'][i:i+1,:], axis=0)


				########### 跟踪的线特征少了，那就补充新的线特征 ###############

				diff_n = self.min_line_cnt - vecline_tracked.shape[0]
				if diff_n > 0:
					if vecline_new.shape[0] >= diff_n:
						for k in range(diff_n):
							vecline_tracked = np.append(vecline_tracked, vecline_new[k:k+1,:], axis=0)
							lineID_tracked.append(lineID_new[k])
							linedescr_tracked = torch.cat((linedescr_tracked, linedescr_new[:,k:k+1,:]), dim=1)
							validpoints_tracked = np.append(validpoints_tracked, validpoints_new[k:k+1,:],axis=0)
					else:
						for k in range(vecline_new.shape[0]):
							vecline_tracked = np.append(vecline_tracked, vecline_new[k:k+1,:], axis=0)
							lineID_tracked.append(lineID_new[k])
							linedescr_tracked = torch.cat((linedescr_tracked, linedescr_new[:,k:k+1,:]), dim=1)
							validpoints_tracked = np.append(validpoints_tracked, validpoints_new[k:k+1,:],axis=0)
							
				self.forwframe_['vecline'] = vecline_tracked
				self.forwframe_['lineID'] = lineID_tracked
				self.forwframe_['linedescriptor'] = linedescr_tracked
				self.forwframe_['valid_points'] = validpoints_tracked

		self.curframe_ = {
				'PointID': self.forwframe_["PointID"].copy(),
				'keyPoint': self.forwframe_["keyPoint"].copy(),
				'vecline': self.forwframe_["vecline"].copy(),
				'lineID': self.forwframe_["lineID"].copy(),
				'pointdescriptor': self.forwframe_["pointdescriptor"].clone(),
				'linedescriptor':  self.forwframe_["linedescriptor"].clone(),
				'valid_points': self.forwframe_["valid_points"].copy(),
				'image': self.forwframe_["image"].copy(),
				}

