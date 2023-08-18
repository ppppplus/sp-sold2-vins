import cv2
import torch
from kornia.feature import SOLD2
from sold2.misc.visualize_util import plot_images, plot_lines
from sold2.model import MyLinefeatureMatchModel

model = SOLD2(pretrained=True)
model = model.cuda()

img0 = cv2.imread('/home/plus/Work/plvins_ws/src/PL-VINS/feature_tracker/scripts/img/img0.JPG', 0)
img1 = cv2.imread('/home/plus/Work/plvins_ws/src/PL-VINS/feature_tracker/scripts/img/img1.JPG', 0)

img0 = (img0 / 255.).astype(float)
img1 = (img1 / 255.).astype(float)

torch_img0 = torch.tensor(img0, dtype=torch.float)[None, None].cuda()
torch_img1 = torch.tensor(img1, dtype=torch.float)[None, None].cuda()
with torch.no_grad():

    out0 = model(torch_img0)
    out1 = model(torch_img1)

# print(out.keys())
# print(dir(model))
line_seg0 = out0["line_segments"][0].cpu().numpy()
line_seg1 = out1["line_segments"][0].cpu().numpy()
desc0 = out0["dense_desc"]
desc1 = out1["dense_desc"]
# desc0 = desc0[None,...]
config = {
    'line_matcher_cfg': {
        'cross_check': True,
        'num_samples': 5,
        'min_dist_pts': 8,
        'top_k_candidates': 10,
        'grid_size': 4
    }
}
match_model = MyLinefeatureMatchModel(config["line_matcher_cfg"])
matches = match_model.match(line_seg0, line_seg1, desc0, desc1)
valid_matches = matches != -1
match_indices = matches[valid_matches]
matched_lines1 = line_seg0[valid_matches][:, :, ::-1]
matched_lines2 = line_seg1[match_indices][:, :, ::-1]

print(matches.shape)
# print(out["dense_desc"].shape)
# print(img.shape)
# plot_images([img, img], ["img", "origin_img"])
# plot_lines([line_seg[:,:,::-1]])