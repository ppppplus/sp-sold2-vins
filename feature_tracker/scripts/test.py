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
for i in range(10):
    my_plextract_model.extract(img0)
    # print(pts.shape, lines.shape)
#%%

# import numpy as np

# # 创建示例原始数组
# original_array = np.random.rand(5, 5, 5)  # 示例原始数组

# # 创建示例索引数组，使其形状与原始数组相同
# index_array = np.empty_like(original_array, dtype=np.int) 

# # 使用循环来构建索引数组
# for i in range(5):
#     for j in range(6):
#         for k in range(9):
#             # 在索引数组的每个位置存储一个索引值
#             index_array[i, j, k] = i  # 这里使用 i 作为示例索引

# # 使用索引数组获取对应的值
# selected_values = original_array[index_array]

# # selected_values 包含了根据索引数组取出的值

# # 打印获取的值的形状
# print(selected_values.shape)
# #%%
# from utils_pl.vision_utils import plot_images, plot_junctions, plot_line_segments, plot_lines
# import numpy as np
# img_pts = plot_junctions(img0, np.flip(pts[0:2]), junc_size=2)
# plot_images([img0, img0], ["img_lines", "img_lines"])
# plot_lines([lines[:, :, ::-1], lines[:, :, ::-1]], ps=3, lw=2)
# plot_images([img_pts], ["img_pts"])
# plot_images([img_pts, img_lines], ["img_pts", "img_lines"])
#%%
import torch
def max_pool(x, nms_radius):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)
def simple_nms(heatmap, nms_radius: int):
        """ Fast Non-maximum suppression to remove nearby points """
        assert(nms_radius >= 0)
        zeros = torch.zeros_like(heatmap)
        max_mask = heatmap == max_pool(heatmap, nms_radius)
        # print(max_pool(heatm?ap, nms_radius))
        
        for _ in range(1):
            # print("max_mask", max_mask)
            supp_mask = max_pool(max_mask.float(), nms_radius) > 0
            # print(supp_mask)
            supp_scores = torch.where(supp_mask, zeros, heatmap)
            # print("score",  supp_scores)
            new_max_mask = supp_scores == max_pool(supp_scores, nms_radius)
            # print(max_pool(supp_scores, nms_radius))
            # print(new_max_mask)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, heatmap, zeros)
# a = torch.arange(1,82)
# a = a.reshape([9,9]).float()
a = torch.randint(0,200,(9,9)).float()
# a = torch.tensor([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]]).float()
b = simple_nms(a[None,...],1)
print(a)
print(b)
# #%%
# import numpy as np
# from utils_pl.vision_utils import plot_images, plot_junctions, plot_line_segments, plot_lines
# img_pts = plot_junctions(img0, np.flip(pts[0:2]), junc_size=1)

# plot_images([img0, img0], ["img_lines", "img_lines"])
# plot_lines([lines[:, :, ::-1], lines[:, :, ::-1]], ps=3, lw=2)
# plot_images([img_pts], ["img_pts"])
# #%%
# # img0 = cv2.imread('/home/plus/Work/plvins_ws/src/PL-VINS/feature_tracker/scripts/img/img0.JPG', 0)
# # img1 = cv2.imread('/home/plus/Work/plvins_ws/src/PL-VINS/feature_tracker/scripts/img/img1.JPG', 0)

# # img0 = (img0 / 255.).astype(float)
# # img1 = (img1 / 255.).astype(float)

# # torch_img0 = torch.tensor(img0, dtype=torch.float)[None, None].cuda()
# # torch_img1 = torch.tensor(img1, dtype=torch.float)[None, None].cuda()
# # with torch.no_grad():

# #     out0 = model(torch_img0)
# #     out1 = model(torch_img1)

# # # print(out.keys())
# # # print(dir(model))
# # line_seg0 = out0["line_segments"][0].cpu().numpy()
# # line_seg1 = out1["line_segments"][0].cpu().numpy()
# # desc0 = out0["dense_desc"]
# # desc1 = out1["dense_desc"]
# # # desc0 = desc0[None,...]
# # config = {
# #     'line_matcher_cfg': {
# #         'cross_check': True,
# #         'num_samples': 5,
# #         'min_dist_pts': 8,
# #         'top_k_candidates': 10,
# #         'grid_size': 4
# #     }
# # }
# # match_model = MyLinefeatureMatchModel(config["line_matcher_cfg"])
# # matches = match_model.match(line_seg0, line_seg1, desc0, desc1)
# # valid_matches = matches != -1
# # match_indices = matches[valid_matches]
# # matched_lines1 = line_seg0[valid_matches][:, :, ::-1]
# # matched_lines2 = line_seg1[match_indices][:, :, ::-1]

# # print(matches.shape)
# # # print(out["dense_desc"].shape)
# # # print(img.shape)
# # # plot_images([img, img], ["img", "origin_img"])
# # # plot_lines([line_seg[:,:,::-1]])
# # %%

# %%
