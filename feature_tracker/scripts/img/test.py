import numpy as np
import matplotlib.pyplot as plt 
import cv2
image0 = np.load("/home/nnplvio_ws/src/sp-sold2-vins/feature_tracker/scripts/img/0_img.npy")
desc0 = np.load("/home/nnplvio_ws/src/sp-sold2-vins/feature_tracker/scripts/img/0_desc.npy")
desc1 = np.load("/home/nnplvio_ws/src/sp-sold2-vins/feature_tracker/scripts/img/1_desc.npy")
desc2 = np.load("/home/nnplvio_ws/src/sp-sold2-vins/feature_tracker/scripts/img/2_desc.npy")
desc01 = np.load("/home/nnplvio_ws/src/sp-sold2-vins/feature_tracker/scripts/img/01_desc.npy")
desc_track = np.load("/home/nnplvio_ws/src/sp-sold2-vins/feature_tracker/scripts/img/desc_tracked.npy")




bf = cv2.BFMatcher(cv2.NORM_HAMMING)

def match(desc1, desc2):
    # desc1 = data["descriptors0"]
    # desc2 = data["descriptors1"]
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))
    pad_width = ((0,0), (0,desc1.shape[1]-desc2.shape[1]))
    desc2  = np.pad(desc2, pad_width, mode="constant", constant_values=0)
    print(desc1.shape, desc2.shape)
    matches = bf.match(desc1.T, desc2.T)
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
            
    '''
        当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
        但有时候最小距离会非常小, 所以设置一个经验值30作为下限。
    '''
    good_match = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_match.append([x.queryIdx, x.trainIdx, x.distance])
    return np.array(good_match).T

match01 = match(desc0, desc1)
print(match01.shape)
num_points = match01.shape[1]
descr_tracked = np.zeros((32,0))
for i in range(num_points):
    id = int(match01[1,i])
    descr_tracked = np.append(descr_tracked, desc1[:,id:id+1], axis=1)
print(descr_tracked.shape)
match12 = match(desc2, np.array(descr_tracked, np.uint8))
print(match12.shape)
# match12 = match(desc1, desc2)
# print(match12.shape)
# # match02 = match(desc0, desc2)
# matcht2 = match(desc0, desc01)
# print(matcht2.shape)
# print(match02.shape)
# match01_2 = match(desc01, desc2)
# vecpoint_new = np.zeros((2,0))
# vecpoint_tracked = np.zeros((2,0))
# pointID_new = []
# pointID_tracked = []
# descr_new = np.zeros((32,0))
# descr_tracked = np.zeros((32,0))
# num_points = match01
# for i in range(num_points):
#     if self.forwframe_['PointID'][i] == -1 :
#         self.forwframe_['PointID'][i] = self.allfeature_cnt
#         self.allfeature_cnt = self.allfeature_cnt+1
#         vecpoint_new = np.append(vecpoint_new, self.forwframe_['keyPoint'][:,i:i+1], axis=1)
#         pointID_new.append(self.forwframe_['PointID'][i])
#         descr_new = np.append(descr_new, self.forwframe_['descriptor'][:,i:i+1], axis=1)
#     else:
#         vecpoint_tracked = np.append(vecpoint_tracked, self.forwframe_['keyPoint'][:,i:i+1], axis=1)
#         pointID_tracked.append(self.forwframe_['PointID'][i])
#         descr_tracked = np.append(descr_tracked, self.forwframe_['descriptor'][:,i:i+1], axis=1)
# plt.imshow(image, cmap="gray")
# plt.title("img")
# plt.show()