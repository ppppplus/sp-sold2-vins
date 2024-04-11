import torch
import numpy as np
import torch.nn.functional as F
import skimage.io
import skimage.transform

from kornia.feature import SOLD2
from ..sold2.misc.geometry_utils import keypoints_to_grid

from .lcnn.config import C, M
from .lcnn.models import hg
from .lcnn.models.line_vectorizer import LineVectorizer
from .lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from .lcnn.postprocess import postprocess
from utils.base_model import BaseExtractModel, BaseMatchModel

class LCNNLineExtractModel(BaseExtractModel):
    def _init(self, params):
        self.device = 'cuda' if torch.cuda.is_available() is True else 'cpu'
        self.params = params
        self.descnet = SOLD2LineDescExtractModel()
        self.net = hg(
            depth=params["depth"],
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out, self.params["head_size"]),
            num_stacks=self.params["num_stacks"],
            num_blocks=self.params["num_blocks"],
            num_classes=sum(sum(self.params["head_size"], [])),
        )
        checkpoint = torch.load(self.params["weight_path"], map_location=self.device)
        self.net = MultitaskLearner(self.net, self.params["head_size"], self.params["loss_weight"])
        self.net = LineVectorizer(self.net, self.params["sampler"])
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net = self.net.to(self.device)
        self.net.eval()
        
    def extract(self, img):
        if img.ndim == 2:
            im = np.repeat(img[:, :, None], 3, 2)
        im = im[:, :, :3]
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - self.params["img_mean"]) / self.params["img_stddev"]
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
        with torch.no_grad():
            input_dict = {
                "image": image.to(self.device),
                "meta": [
                    {
                        "junc": torch.zeros(1, 2).to(self.device),
                        "jtyp": torch.zeros(1, dtype=torch.uint8).to(self.device),
                        "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                        "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                    }
                ],
                "target": {
                    "jmap": torch.zeros([1, 1, 128, 128]).to(self.device),
                    "joff": torch.zeros([1, 1, 2, 128, 128]).to(self.device),
                },
                "mode": "testing",
            }
            H = self.net(input_dict)["preds"]

        lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
        scores = H["score"][0].cpu().numpy()
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)
        desc, valid_points = self.descnet.desc_extract(img, nlines)

        return nlines, desc, valid_points # num_lines*2*2; 128*num_lines*num_samples; num_lines*5
    
class SOLD2LineDescExtractModel():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() is True else 'cpu'
        self.net = SOLD2(pretrained=True)
        self.net = self.net.to(self.device)
        self.sampling_mode = "regular"
        self.num_samples = 5
        self.min_dist_pts = 8
        # self.grid_size = params["grid_size"]
        self.line_score = False
        
    def desc_extract(self, img, vecline):
        img = (img / 255.).astype(float)
        img_size = img.shape    # 480*752
        torch_img = torch.tensor(img, dtype=torch.float)[None, None].to(self.device)
        with torch.no_grad():
            out = self.net(torch_img)
        # vecline= out["line_segments"][0].cpu().numpy()  
        desc = out["dense_desc"] # 1*128*img_size/grid_size
        # 参照line_matching中的函数做描述子和线之间的对应
        if self.sampling_mode == "regular":
            line_points, valid_points = self.sample_line_points(vecline)  
        else:
            line_points, valid_points = self.sample_salient_points(
                vecline, desc, img_size, self.sampling_mode)

        line_points = torch.tensor(line_points.reshape(-1, 2),
                                    dtype=torch.float, device=self.device)
        # Extract the descriptors for each point
        grid = keypoints_to_grid(line_points, img_size)
        desc = F.normalize(F.grid_sample(desc, grid)[0, :, :, 0], dim=0)
        desc = desc.reshape((-1, len(vecline), self.num_samples)) # reshape为每个线段对应的desc,128*num_lines*num_samples

        return desc, valid_points  # 128*num_lines*num_samples; num_lines*5
    
    def sample_salient_points(self, line_seg, desc, img_size,
                              saliency_type='d2_net'):
        """
        Sample the most salient points along each line segments, with a
        minimal distance between each point. Pad the remaining points.
        Inputs:
            line_seg: an Nx2x2 torch.Tensor.
            desc: a NxDxHxW torch.Tensor.
            image_size: the original image size.
            saliency_type: 'd2_net' or 'asl_feat'.
        Outputs:
            line_points: an Nxnum_samplesx2 np.array.
            valid_points: a boolean Nxnum_samples np.array.
        """
        if not self.line_score:
            # Compute the score map
            if saliency_type == "d2_net":
                score = self.d2_net_saliency_score(desc)
            else:
                score = self.asl_feat_saliency_score(desc)

        num_lines = len(line_seg)
        line_lengths = np.linalg.norm(line_seg[:, 0] - line_seg[:, 1], axis=1)

        # The number of samples depends on the length of the line
        num_samples_lst = np.clip(line_lengths // self.min_dist_pts,
                                  2, self.num_samples)
        line_points = np.empty((num_lines, self.num_samples, 2), dtype=float)
        valid_points = np.empty((num_lines, self.num_samples), dtype=bool)

        # Sample the score on a fixed number of points of each line
        n_samples_per_region = 4
        for n in np.arange(2, self.num_samples + 1):
            sample_rate = n * n_samples_per_region
            # Consider all lines where we can fit up to n points
            cur_mask = num_samples_lst == n
            cur_line_seg = line_seg[cur_mask]
            cur_num_lines = len(cur_line_seg)
            if cur_num_lines == 0:
                continue
            line_points_x = np.linspace(cur_line_seg[:, 0, 0],
                                        cur_line_seg[:, 1, 0],
                                        sample_rate, axis=-1)
            line_points_y = np.linspace(cur_line_seg[:, 0, 1],
                                        cur_line_seg[:, 1, 1],
                                        sample_rate, axis=-1)
            cur_line_points = np.stack([line_points_x, line_points_y],
                                       axis=-1).reshape(-1, 2)
            # cur_line_points is of shape (n_cur_lines * sample_rate, 2)
            cur_line_points = torch.tensor(cur_line_points, dtype=torch.float,
                                           device=self.device)
            grid_points = keypoints_to_grid(cur_line_points, img_size)

            if self.line_score:
                # The saliency score is high when the activation are locally
                # maximal along the line (and not in a square neigborhood)
                line_desc = F.grid_sample(desc, grid_points).squeeze()
                line_desc = line_desc.reshape(-1, cur_num_lines, sample_rate)
                line_desc = line_desc.permute(1, 0, 2)
                if saliency_type == "d2_net":
                    scores = self.d2_net_saliency_score(line_desc)
                else:
                    scores = self.asl_feat_saliency_score(line_desc)
            else:
                scores = F.grid_sample(score.unsqueeze(1),
                                       grid_points).squeeze()

            # Take the most salient point in n distinct regions
            scores = scores.reshape(-1, n, n_samples_per_region)
            best = torch.max(scores, dim=2, keepdim=True)[1].cpu().numpy()
            cur_line_points = cur_line_points.reshape(-1, n,
                                                      n_samples_per_region, 2)
            cur_line_points = np.take_along_axis(
                cur_line_points, best[..., None], axis=2)[:, :, 0]

            # Pad
            cur_valid_points = np.ones((cur_num_lines, self.num_samples),
                                       dtype=bool)
            cur_valid_points[:, n:] = False
            cur_line_points = np.concatenate([
                cur_line_points,
                np.zeros((cur_num_lines, self.num_samples - n, 2), dtype=float)],
                axis=1)

            line_points[cur_mask] = cur_line_points
            valid_points[cur_mask] = cur_valid_points

        return line_points, valid_points

    def sample_line_points(self, line_seg):
        """
        Regularly sample points along each line segments, with a minimal
        distance between each point. Pad the remaining points.
        Inputs:
            line_seg: an Nx2x2 torch.Tensor.
        Outputs:
            line_points: an Nxnum_samplesx2 np.array.
            valid_points: a boolean Nxnum_samples np.array.
        """
        num_lines = len(line_seg)
        line_lengths = np.linalg.norm(line_seg[:, 0] - line_seg[:, 1], axis=1)

        # Sample the points separated by at least min_dist_pts along each line
        # The number of samples depends on the length of the line
        num_samples_lst = np.clip(line_lengths // self.min_dist_pts,
                                  2, self.num_samples)
        line_points = np.empty((num_lines, self.num_samples, 2), dtype=float)
        valid_points = np.empty((num_lines, self.num_samples), dtype=bool)
        for n in np.arange(2, self.num_samples + 1):
            # Consider all lines where we can fit up to n points
            cur_mask = num_samples_lst == n
            cur_line_seg = line_seg[cur_mask]
            line_points_x = np.linspace(cur_line_seg[:, 0, 0],
                                        cur_line_seg[:, 1, 0],
                                        n, axis=-1)
            line_points_y = np.linspace(cur_line_seg[:, 0, 1],
                                        cur_line_seg[:, 1, 1],
                                        n, axis=-1)
            cur_line_points = np.stack([line_points_x, line_points_y], axis=-1)

            # Pad
            cur_num_lines = len(cur_line_seg)
            cur_valid_points = np.ones((cur_num_lines, self.num_samples),
                                       dtype=bool)
            cur_valid_points[:, n:] = False
            cur_line_points = np.concatenate([
                cur_line_points,
                np.zeros((cur_num_lines, self.num_samples - n, 2), dtype=float)],
                axis=1)

            line_points[cur_mask] = cur_line_points
            valid_points[cur_mask] = cur_valid_points

        return line_points, valid_points