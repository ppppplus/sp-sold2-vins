#%% 
import cv2
import torch
# from kornia.feature import SOLD2
# from sold2.misc.visualize_util import plot_images, plot_lines
# from sold2.model import MyLinefeatureMatchModel
import yaml
from utils_pl.my_pl_model import create_plextract_instance

# model = SOLD2(pretrained=True)
# model = model.cuda()
####### create pl model 
yamlPath = "/home/nvidia/Work/sp-sold2-vins_ws/src/sp-sold2-vins/config/feature_tracker/sp-sold2_config.yaml"
with open(yamlPath,'rb') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    pl_params = params["pl_feature_cfg"]
    point_params = params["point_feature_cfg"]
    line_params = params["line_feature_cfg"]

    camera_params = params["camera_cfg"]

my_plextract_model = create_plextract_instance(pl_params)

img0 = cv2.imread('/home/nvidia/Documents/terrace0.JPG', 0)
print(img0.shape)
#%%
# img1 = cv2.imread('/home/plus/Work/plvins_ws/src/PL-VINS/feature_tracker/scripts/img/img1.JPG', 0)

# img0 = (img0 / 255.).astype(float)
pts, pts_desc, lines, lines_desc, valid_points = my_plextract_model.extract(img0)
#%%
import numpy as np
from utils_pl.vision_utils import plot_images, plot_junctions, plot_line_segments, plot_lines
img_pts = plot_junctions(img0, np.flip(pts[0:2]), junc_size=1)

plot_images([img0, img0], ["img_lines", "img_lines"])
plot_lines([lines[:, :, ::-1], lines[:, :, ::-1]], ps=3, lw=2)
plot_images([img_pts], ["img_pts"])
#%%
# img0 = cv2.imread('/home/plus/Work/plvins_ws/src/PL-VINS/feature_tracker/scripts/img/img0.JPG', 0)
# img1 = cv2.imread('/home/plus/Work/plvins_ws/src/PL-VINS/feature_tracker/scripts/img/img1.JPG', 0)

# img0 = (img0 / 255.).astype(float)
# img1 = (img1 / 255.).astype(float)

# torch_img0 = torch.tensor(img0, dtype=torch.float)[None, None].cuda()
# torch_img1 = torch.tensor(img1, dtype=torch.float)[None, None].cuda()
# with torch.no_grad():

#     out0 = model(torch_img0)
#     out1 = model(torch_img1)

# # print(out.keys())
# # print(dir(model))
# line_seg0 = out0["line_segments"][0].cpu().numpy()
# line_seg1 = out1["line_segments"][0].cpu().numpy()
# desc0 = out0["dense_desc"]
# desc1 = out1["dense_desc"]
# # desc0 = desc0[None,...]
# config = {
#     'line_matcher_cfg': {
#         'cross_check': True,
#         'num_samples': 5,
#         'min_dist_pts': 8,
#         'top_k_candidates': 10,
#         'grid_size': 4
#     }
# }
# match_model = MyLinefeatureMatchModel(config["line_matcher_cfg"])
# matches = match_model.match(line_seg0, line_seg1, desc0, desc1)
# valid_matches = matches != -1
# match_indices = matches[valid_matches]
# matched_lines1 = line_seg0[valid_matches][:, :, ::-1]
# matched_lines2 = line_seg1[match_indices][:, :, ::-1]

# print(matches.shape)
# # print(out["dense_desc"].shape)
# # print(img.shape)
# # plot_images([img, img], ["img", "origin_img"])
# # plot_lines([line_seg[:,:,::-1]])
# %%
