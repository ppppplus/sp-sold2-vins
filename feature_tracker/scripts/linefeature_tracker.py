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


class LineFeatureTracker:
	def __init__(self, extract_model, match_model, cams, num_samples=8, min_cnt=150):
		# extract_model为自定义点特征模型类，其中提供extract方法接受一个图像输入，输出线特征信息（lines, lens）
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.extractor = extract_model
		self.matcher = match_model
		self.num_samples = num_samples
		# frame_仿照源码中的FrameLines，含有frame_id, img, vecline, lineID, keylsd, lbd_descr六个属性
		self.forwframe_ = {
				'frame_id': None, 
				'vecline': np.zeros((0,2,2)),
				'lineID': None,
				# 'keylsd': None,
				'descriptor': torch.zeros((128,0)).to(self.device),
				'valid_points': None,	# 存储哪些线采样点为有效的
				'image': None,
				}
		self.curframe_ = {
				'frame_id': None, 
				'vecline': np.zeros((0,2,2)),
				'lineID': None,
				# 'keylsd': None,
				'descriptor': torch.zeros((128,0)).to(self.device),
				'valid_points': None,	# 存储哪些线采样点为有效的
				'image': None,
				}
		# self.forwframe_ = {
		# 		'frame_id': None, 
		# 		'vecline': [],
		# 		'lineID': None,
		# 		# 'keylsd': None,
		# 		'descriptor': np.zeros((128,0)),
		# 		'image': None,
		# 		}
	
		self.camera = cams
		self.new_frame = None
		self.allfeature_cnt = 0
		self.min_cnt = min_cnt
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
		if not self.forwframe_['lineID']:
			# 初始化第一帧图像
			self.forwframe_['lineID'] = []
			self.forwframe_['image'] = self.new_frame
			self.curframe_['image'] = self.new_frame
			first_image_flag = True
		else:
			self.forwframe_['lineID'] = []
			self.forwframe_['descriptor'] = torch.zeros((128,0)).to(self.device)
			self.forwframe_['valid_points'] = None
			self.forwframe_['image'] = self.new_frame	# 建立新的帧

		# TODO: 利用线特征提取器提取new_frame的特征信息
		print('*'*10 + " current frame " + '*'*10)
		start_time = time()
		self.forwframe_['vecline'], self.forwframe_['descriptor'], self.forwframe_['valid_points']  = self.extractor.extract(self.new_frame)	# vecline为num_line*2*2，desc为128*num_line*num_samples

		global run_time
		run_time += ( time()-start_time )
		print("total run time is :", run_time)

		lines_num = self.forwframe_['vecline'].shape[0]
		print("current number of lines is :", lines_num)
		

		# if keyPoint_size < self.min_cnt-50:
		# 	self.forwframe_['keyPoint'], self.forwframe_['descriptor'], heatmap = self.SuperPoint_Ghostnet.run(self.new_frame, conf_thresh=0.01)
		# 	keyPoint_size = self.forwframe_['keyPoint'].shape[1]
		# 	print("next keypoint size is ", keyPoint_size)

		

		for _ in range(lines_num):
			if first_image_flag == True:
				self.forwframe_['lineID'].append(self.allfeature_cnt)
				self.allfeature_cnt = self.allfeature_cnt+1
			else:
				self.forwframe_['lineID'].append(-1)
		
		##################### 开始处理匹配的线特征 ###############################
		if self.curframe_['vecline'].shape[0] > 0:
			start_time = time()
			index_lines1, index_lines2 = self.matcher.match( 
									self.forwframe_['vecline'],
									self.curframe_['vecline'],
									self.forwframe_['descriptor'][None,...], 
									self.curframe_['descriptor'][None,...],
									self.forwframe_['valid_points'],
									self.curframe_['valid_points']
							)
			# print("index:", index_lines1, index_lines2)
			global match_time
			match_time += time()-start_time
			print("match time is :", match_time)
			print("match size is :", index_lines1.shape[0])
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

		# if not self.no_display :	
		# 	out1 = (np.dstack((self.curframe_['image'], self.curframe_['image'], self.curframe_['image'])) * 255.).astype('uint8')
		# 	for i in range(len(self.curframe_['PointID'])):
		# 		pts1 = (int(round(self.curframe_['keyPoint'][0,i]))-3, int(round(self.curframe_['keyPoint'][1,i]))-3)
		# 		pts2 = (int(round(self.curframe_['keyPoint'][0,i]))+3, int(round(self.curframe_['keyPoint'][1,i]))+3)
		# 		pt2 = (int(round(self.curframe_['keyPoint'][0,i])), int(round(self.curframe_['keyPoint'][1,i])))
		# 		cv2.rectangle(out1, pts1, pts2, (0,255,0))
		# 		cv2.circle(out1, pt2, 2, (255, 0, 0), -1)
		# 		# cv2.putText(out1, str(self.curframe_['PointID'][i]), pt2, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX , 0.3, (0, 0, 255), lineType=5)
		# 	cv2.putText(out1, 'pre_image Point', (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), lineType=16)

		# 	out2 = (np.dstack((self.forwframe_['image'], self.forwframe_['image'], self.forwframe_['image'])) * 255.).astype('uint8')
		# 	for i in range(len(self.forwframe_['PointID'])):
		# 		pts1 = (int(round(self.forwframe_['keyPoint'][0,i]))-3, int(round(self.forwframe_['keyPoint'][1,i]))-3)
		# 		pts2 = (int(round(self.forwframe_['keyPoint'][0,i]))+3, int(round(self.forwframe_['keyPoint'][1,i]))+3)
		# 		pt2 = (int(round(self.forwframe_['keyPoint'][0,i])), int(round(self.forwframe_['keyPoint'][1,i])))
		# 		cv2.rectangle(out2, pts1, pts2, (0,255,0))
		# 		cv2.circle(out2, pt2, 2, (0, 0, 255), -1)
		# 		# cv2.putText(out2, str(self.forwframe_['PointID'][i]), pt2, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 0, 255), lineType=5)
		# 	cv2.putText(out2, 'cur_image Point', (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), lineType=16)

		# 	min_conf = 0.001
		# 	heatmap[heatmap < min_conf] = min_conf
		# 	heatmap = -np.log(heatmap)
		# 	heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
		# 	out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
		# 	out3 = (out3*255).astype('uint8')
		# 	# print()
		# 	print(out1.shape, out2.shape, out3.shape)
		# 	out = np.hstack((out1, out2, out3))
			
		# 	out = cv2.resize(out, (3*self.width, self.height))

		# 	cv2.namedWindow("feature detector window",1)
		# 	# cv2.resizeWindow("feature detector window", 640*3, 480)
		# 	cv2.imshow('feature detector window',out)
		# 	cv2.waitKey(1)

		self.curframe_ = copy.deepcopy(self.forwframe_)

