"""
本文件定义了一个类来实现线特征提取的功能，替代PL-VINS源码中的linefeature_tracker.cpp
"""
import cv2
import copy
import rospy
import numpy as np 
import torch
from time import time

# from utils.PointTracker import PointTracker
# from utils.feature_process import SuperPointFrontend_torch, SuperPointFrontend
# run_time = 0.0
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


class LineFeatureTracker:
	def __init__(self, extract_model, match_model, cams, num_samples=8, min_cnt=150, opencv=False):
		# extract_model为自定义点特征模型类，其中提供extract方法接受一个图像输入，输出线特征信息（lines, lens）
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.extractor = extract_model
		self.matcher = match_model
		self.num_samples = num_samples
		# frame_仿照源码中的FrameLines，含有frame_id, img, vecline, lineID, keylsd, lbd_descr六个属性
		
		self.forwframe_ = {
				'lineID': None,
				'vecline' : np.zeros((0,2,2)),
				}
		self.curframe_ = {
				'lineID': None,
				'vecline' : np.zeros((0,2,2)),
				}
	
		self.camera = cams
		self.new_frame = None
		self.allfeature_cnt = 0
		self.min_cnt = min_cnt
		self.no_display = True
		self.opencv = opencv
		
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
		self.new_frame = self.camera.undistortImg(new_img)
		self.first_image_flag = False
		if self.opencv:
			self.processcvImage()
		else:
			self.processImage()

	def processImage(self):
		if not self.forwframe_['lineID']:
			# 初始化第一帧图像
			self.forwframe_['lineID'] = []
			self.forwframe_['image'] = self.new_frame
			self.curframe_['image'] = self.new_frame
			self.first_image_flag = True
		else:
			self.forwframe_['lineID'] = []
			self.forwframe_['descriptor'] = torch.zeros((128,0)).to(self.device)
			self.forwframe_['valid_points'] = None
			self.forwframe_['image'] = self.new_frame	# 建立新的帧

		# 利用线特征提取器提取new_frame的特征信息
		print('*'*10 + " current frame " + '*'*10)
		start_time = time()
		self.forwframe_['vecline'], self.forwframe_['descriptor'], self.forwframe_['valid_points']  = self.extractor.extract(self.new_frame)	# vecline为num_line*2*2，desc为128*num_line*num_samples

		# global run_time
		# run_time += ( time()-start_time )
		# print("total run time is :", run_time)
		print("line extraction time is:", time()-start_time)
		lines_num = self.forwframe_['vecline'].shape[0]
		print("current number of lines is :", lines_num)
		

		# if keyPoint_size < self.min_cnt-50:
		# 	self.forwframe_['keyPoint'], self.forwframe_['descriptor'], heatmap = self.SuperPoint_Ghostnet.run(self.new_frame, conf_thresh=0.01)
		# 	keyPoint_size = self.forwframe_['keyPoint'].shape[1]
		# 	print("next keypoint size is ", keyPoint_size)

		

		for _ in range(lines_num):
			if self.first_image_flag == True:
				self.forwframe_['lineID'].append(self.allfeature_cnt)
				self.allfeature_cnt = self.allfeature_cnt+1
			else:
				self.forwframe_['lineID'].append(-1)
		
		##################### 开始处理匹配的线特征 ###############################
		if self.curframe_['vecline'].shape[0] > 0:
			start_time = time()
			index_lines1, index_lines2 = self.matcher.match( 
									{
										"vecline0": self.forwframe_['vecline'],
										"vecline1": self.curframe_['vecline'],
										"descriptor0": self.forwframe_['descriptor'][None,...], 
										"descriptor1": self.curframe_['descriptor'][None,...],
										"valid0": self.forwframe_['valid_points'],
										"valid1": self.curframe_['valid_points']
									}
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
			descr_new = torch.zeros((128,0,self.num_samples)).to(self.device)
			descr_tracked = torch.zeros((128,0,self.num_samples)).to(self.device)

			for i in range(lines_num):
				if self.forwframe_['lineID'][i] == -1 :	# -1表示当前ID对应的line没有track到
					self.forwframe_['lineID'][i] = self.allfeature_cnt	# 没有跟踪到的线则编号为新的
					self.allfeature_cnt = self.allfeature_cnt+1
					vecline_new = np.append(vecline_new, self.forwframe_['vecline'][i:i+1,...], axis=0)	# 取出没有跟踪到的线信息并放入下一帧
					lineID_new.append(self.forwframe_['lineID'][i])
					descr_new = torch.cat((descr_new, self.forwframe_['descriptor'][:,i:i+1,:]), dim=1)
					validpoints_new = np.append(validpoints_new, self.forwframe_['valid_points'][i:i+1,:], axis=0)
				else:
					# 当前line已被track
					lineID_tracked.append(self.forwframe_['lineID'][i])
					vecline_tracked = np.append(vecline_tracked, self.forwframe_['vecline'][i:i+1,...], axis=0)
					descr_tracked = torch.cat((descr_tracked, self.forwframe_['descriptor'][:,i:i+1,:]), dim=1)
					validpoints_tracked = np.append(validpoints_tracked, self.forwframe_['valid_points'][i:i+1,:], axis=0)


			########### 跟踪的线特征少了，那就补充新的线特征 ###############

			diff_n = self.min_cnt - vecline_tracked.shape[0]
			if diff_n > 0:
				if vecline_new.shape[0] >= diff_n:
					for k in range(diff_n):
						vecline_tracked = np.append(vecline_tracked, vecline_new[k:k+1,:], axis=0)
						lineID_tracked.append(lineID_new[k])
						descr_tracked = torch.cat((descr_tracked, descr_new[:,k:k+1,:]), dim=1)
						validpoints_tracked = np.append(validpoints_tracked, validpoints_new[k:k+1,:],axis=0)
				else:
					for k in range(vecline_new.shape[0]):
						vecline_tracked = np.append(vecline_tracked, vecline_new[k:k+1,:], axis=0)
						lineID_tracked.append(lineID_new[k])
						descr_tracked = torch.cat((descr_tracked, descr_new[:,k:k+1,:]), dim=1)
						validpoints_tracked = np.append(validpoints_tracked, validpoints_new[k:k+1,:],axis=0)
						
			self.forwframe_['vecline'] = vecline_tracked
			self.forwframe_['lineID'] = lineID_tracked
			self.forwframe_['descriptor'] = descr_tracked
			self.forwframe_['valid_points'] = validpoints_tracked

		self.curframe_ = copy.deepcopy(self.forwframe_)

	def processcvImage(self):		
		if not self.forwframe_['lineID']:
			# 初始化第一帧图像
			self.forwframe_['lineID'] = []
			self.forwframe_['image'] = self.new_frame
			self.curframe_['image'] = self.new_frame
			self.first_image_flag = True
		else:
			self.forwframe_['lineID'] = []
			self.forwframe_['descriptor'] = []
			self.forwframe_['image'] = self.new_frame	# 建立新的帧

		# 利用线特征提取器提取new_frame的特征信息
		print('*'*10 + " current frame " + '*'*10)
		start_time = time()
		self.forwframe_['vecline'], self.forwframe_['descriptor'] = self.extractor.extract(self.new_frame)	# num_line*2*2，desc_dim*num_line
		# global run_time
		# run_time += ( time()-start_time )
		# print("total run time is :", run_time)
		print("line extraction time is:", time()-start_time)
		lines_num = self.forwframe_['vecline'].shape[0]
		desc_dim = self.forwframe_['descriptor'].shape[0]
		print("current number of lines is :", lines_num)
	
		for _ in range(lines_num):
			if self.first_image_flag == True:
				self.forwframe_['lineID'].append(self.allfeature_cnt)
				self.allfeature_cnt = self.allfeature_cnt+1
			else:
				self.forwframe_['lineID'].append(-1)
		
		##################### 开始处理匹配的线特征 ###############################
		if self.curframe_['vecline'].shape[0] > 0:
			start_time = time()
			matches_index = self.matcher.match( 
									{
										"descriptors0": self.forwframe_['descriptor'], 
										"descriptors1": self.curframe_['descriptor'],
									}
							)
			# print("index:", index_lines1, index_lines2)
			print("line match time is :", time()-start_time)
			print("line match size is :", matches_index.shape[0])
			######################## 保证匹配得到的lineID相同 #####################
			for k in range(matches_index.shape[0]):
				self.forwframe_['lineID'][matches_index[k,0]] = self.curframe_['lineID'][matches_index[k,1]]

			################### 将跟踪的线与没跟踪的线进行区分 #####################
			vecline_new = np.zeros((0,2,2))
			vecline_tracked = np.zeros((0,2,2))
			lineID_new = []
			lineID_tracked = []
			descr_new = np.zeros((desc_dim,0))
			descr_tracked = np.zeros((desc_dim,0))

			for i in range(lines_num):
				if self.forwframe_['lineID'][i] == -1 :	# -1表示当前ID对应的line没有track到
					self.forwframe_['lineID'][i] = self.allfeature_cnt	# 没有跟踪到的线则编号为新的
					self.allfeature_cnt = self.allfeature_cnt+1
					vecline_new = np.append(vecline_new, self.forwframe_['vecline'][i:i+1,...], axis=0)	# 取出没有跟踪到的线信息并放入下一帧
					lineID_new.append(self.forwframe_['lineID'][i])
					descr_new = np.append(descr_new, self.forwframe_['descriptor'][:,i:i+1], axis=1)
				else:
					# 当前line已被track
					lineID_tracked.append(self.forwframe_['lineID'][i])
					vecline_tracked = np.append(vecline_tracked, self.forwframe_['vecline'][i:i+1,...], axis=0)
					descr_tracked = np.append(descr_tracked, self.forwframe_['descriptor'][:,i:i+1], axis=1)

			########### 跟踪的线特征少了，那就补充新的线特征 ###############

			diff_n = self.min_cnt - vecline_tracked.shape[0]
			if diff_n > 0:
				if vecline_new.shape[0] >= diff_n:
					for k in range(diff_n):
						vecline_tracked = np.append(vecline_tracked, vecline_new[k:k+1,:], axis=0)
						lineID_tracked.append(lineID_new[k])
						descr_tracked = np.append(descr_tracked, descr_new[:,k:k+1], axis=1)
				else:
					for k in range(vecline_new.shape[0]):
						vecline_tracked = np.append(vecline_tracked, vecline_new[k:k+1,:], axis=0)
						lineID_tracked.append(lineID_new[k])
						descr_tracked = np.append(descr_tracked, descr_new[:,k:k+1], axis=1)
						
			self.forwframe_['vecline'] = vecline_tracked
			self.forwframe_['lineID'] = lineID_tracked
			self.forwframe_['descriptor'] = descr_tracked

		self.curframe_ = copy.deepcopy(self.forwframe_)

