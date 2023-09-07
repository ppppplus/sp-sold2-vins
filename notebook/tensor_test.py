import torch 
from pyinstrument import Profiler
import numpy as np
# cand_vecs = torch.load("/home/nvidia/Work/sp-sold2-vins_ws/src/sp-sold2-vins/notebook/tensor/cand_vecs.pt")
# dir_vecs = torch.load("/home/nvidia/Work/sp-sold2-vins_ws/src/sp-sold2-vins/notebook/tensor/dir_vecs.pt")
# line_dists = torch.load("/home/nvidia/Work/sp-sold2-vins_ws/src/sp-sold2-vins/notebook/tensor/line_dists.pt")
# cand_vecs_norm = torch.load("/home/nvidia/Work/sp-sold2-vins_ws/src/sp-sold2-vins/notebook/tensor/cand_vecs_norm.pt")


# profiler = Profiler()
# for i in range(10):
#     profiler.start()
#     proj = (torch.einsum('bij,bjk->bik', cand_vecs, dir_vecs[..., None])
#                     / line_dists[..., None, None])
#     # proj is num_segs x num_junction x 1
#     # proj_mask = (proj >=0) * (proj <= 1)
#     # cand_angles = torch.acos(
#     #     torch.einsum('bij,bjk->bik', cand_vecs, dir_vecs[..., None])
#     #     / cand_vecs_norm[..., None])
#     profiler.stop()
#     profiler.print()
heatmap2 = np.load("/home/nvidia/Work/sp-sold2-vins_ws/src/sp-sold2-vins/notebook/tensor/heatmap2.npy")
# junctions2 = np.load("/home/nvidia/Work/sp-sold2-vins_ws/src/sp-sold2-vins/notebook/tensor/junctions2.npy")
# img2 = np.load("/home/nvidia/Work/sp-sold2-vins_ws/src/sp-sold2-vins/notebook/tensor/img.npy")
# plot_images([img], ["Img"])
# print(junctions1.shape, junctions2.shape)
# img_junctions2 = plot_junctions(img2, junctions2.squeeze(), junc_size=1)
# plot_images([img_junctions1, img_junctions2], ["IMG1", "IMG2"])
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
 
# 概率霍夫变换
# src = cv.imread("E:\\Hough.png")
# img = src.copy()
# gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = heatmap2.copy()*255
img = img.astype(np.uint8)
# print(img.shape)
# print(img)
profiler = Profiler()
profiler.start()
dst_img = cv.Canny(img, 20, 50)

lines = cv.HoughLinesP(dst_img, 1, np.pi / 180, 20)
profiler.stop()
profiler.print()
# for x1, y1, x2, y2 in lines[0]:
#     cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
 
# cv.imshow("HoughLinesP", img)
# cv.waitKey(0)
