import torch
import numpy as np
import torch.nn.functional as F
import skimage.io
import skimage.transform

import lcnn
from lcnn.config import C, M
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from lcnn.postprocess import postprocess
from utils.base_model import BaseExtractModel, BaseMatchModel

class LCNNLineExtractModel(BaseExtractModel):
    def _init(self, params):
        self.device = 'cuda' if torch.cuda.is_available() is True else 'cpu'
        self.params = params
        self.net = lcnn.models.hg(
            depth=params["depth"],
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=self.params["num_stacks"],
            num_blocks=self.params["num_blocks"],
            num_classes=sum(sum(self.params["head_size"], [])),
        )
        checkpoint = torch.load(self.params["weight_path"], map_location=self.device)
        self.net = MultitaskLearner(self.net, self.)
        self.net = LineVectorizer(self.net)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net = self.net.to(self.device)
        self.net.eval()
        
    def extract(self, img):
        im_resized = skimage.transform.resize(img, (512, 512)) * 255
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

        lines = H["lines"][0].cpu().numpy() / 128 * img.shape[:2]
        scores = H["score"][0].cpu().numpy()
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (img.shape[0] ** 2 + img.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

        return nlines # num_lines*2*2; 128*num_lines*num_samples; num_lines*5
    

class WunschLinefeatureMatchModel(BaseMatchModel):
    def _init(self, params):
        # self.net = WunschLineMatcher(**line_matcher_cfg)
        self.cross_check = params["cross_check"]
        self.num_samples = params["num_samples"]
        # self.min_dist_pts = params["min_dist_pts"]
        self.top_k_candidates = params["top_k_candidates"]
        # self.grid_size = params["grid_size"]
        self.line_score = params["line_score"]  # True to compute saliency on a line
        # self.sampling_mode = params["sampling"]
        # if self.sampling_mode not in ["regular", "d2_net", "asl_feat"]:
        #     raise ValueError("Wrong sampling mode: " + self.sampling_mode)

    def match(self, vecline1, vecline2, desc1, desc2, valid1, valid2):
        # matches = self.net.forward(vecline1, vecline2, desc1, desc2)
        # index_lines1 = np.arange(vecline1.shape[0])
        # valid_matches = matches != -1
        # index_lines1 = index_lines1[valid_matches][:, :, ::-1]
        # index_lines2 = matches[valid_matches]
        # # matched_lines2 = vecline2[match_indices][:, :, ::-1]
        # return index_lines1, index_lines2
        # device = desc1.device

        # Default case when an image has no lines
        if len(vecline1) == 0:
            return np.empty((0), dtype=int)
        if len(vecline2) == 0:
            return -np.ones(len(vecline1), dtype=int)
        desc1 = desc1.reshape((128,-1))
        desc2 = desc2.reshape((128,-1))


        # Precompute the distance between line points for every pair of lines
        # Assign a score of -1 for unvalid points
        scores = desc1.t() @ desc2
        scores[~valid1.flatten()] = -1
        scores[:, ~valid2.flatten()] = -1
        scores = scores.reshape(len(vecline1), self.num_samples,
                                len(vecline2), self.num_samples)
        scores = scores.permute(0, 2, 1, 3)
        # scores.shape = (n_lines1, n_lines2, num_samples, num_samples)，也就是每一对线段之间的匹配矩阵

        # Pre-filter the line candidates and find the best match for each line
        matches = self.filter_and_match_lines(scores)

        # [Optionally] filter matches with mutual nearest neighbor filtering
        if self.cross_check:
            matches2 = self.filter_and_match_lines(
                scores.permute(1, 0, 3, 2))
            mutual = matches2[matches] == np.arange(len(vecline1))
            matches[~mutual] = -1
        index_lines1 = np.arange(vecline1.shape[0])
        valid_matches = matches != -1
        index_lines1 = index_lines1[valid_matches]
        index_lines2 = matches[valid_matches]
        # matched_lines2 = vecline2[match_indices][:, :, ::-1]
        return index_lines1, index_lines2
    
    def filter_and_match_lines(self, scores):
        """
        Use the scores to keep the top k best lines, compute the Needleman-
        Wunsch algorithm on each candidate pairs, and keep the highest score.
        Inputs:
            scores: a (N, M, n, n) torch.Tensor containing the pairwise scores
                    of the elements to match.
        Outputs:
            matches: a (N) np.array containing the indices of the best match
        """
        # Pre-filter the pairs and keep the top k best candidate lines
        line_scores1 = scores.max(3)[0]
        valid_scores1 = line_scores1 != -1
        line_scores1 = ((line_scores1 * valid_scores1).sum(2)
                        / valid_scores1.sum(2))
        line_scores2 = scores.max(2)[0]
        valid_scores2 = line_scores2 != -1
        line_scores2 = ((line_scores2 * valid_scores2).sum(2)
                        / valid_scores2.sum(2))
        line_scores = (line_scores1 + line_scores2) / 2
        topk_lines = torch.argsort(line_scores,
                                dim=1)[:, -self.top_k_candidates:]
        scores, topk_lines = scores.cpu().numpy(), topk_lines.cpu().numpy()
        # topk_lines.shape = (n_lines1, top_k_candidates)
        top_scores = np.take_along_axis(scores, topk_lines[:, :, None, None],
                                        axis=1)

        # Consider the reversed line segments as well
        top_scores = np.concatenate([top_scores, top_scores[..., ::-1]],
                                    axis=1)

        # Compute the line distance matrix with Needleman-Wunsch algo and
        # retrieve the closest line neighbor
        n_lines1, top2k, n, m = top_scores.shape
        top_scores = top_scores.reshape(n_lines1 * top2k, n, m)
        nw_scores = self.needleman_wunsch(top_scores)
        nw_scores = nw_scores.reshape(n_lines1, top2k)
        matches = np.mod(np.argmax(nw_scores, axis=1), top2k // 2)
        matches = topk_lines[np.arange(n_lines1), matches]
        return matches

    def needleman_wunsch(self, scores):
        """
        Batched implementation of the Needleman-Wunsch algorithm.
        The cost of the InDel operation is set to 0 by subtracting the gap
        penalty to the scores.
        Inputs:
            scores: a (B, N, M) np.array containing the pairwise scores
                    of the elements to match.
        """
        b, n, m = scores.shape

        # Recalibrate the scores to get a gap score of 0
        gap = 0.1
        nw_scores = scores - gap

        # Run the dynamic programming algorithm
        nw_grid = np.zeros((b, n + 1, m + 1), dtype=float)
        for i in range(n):
            for j in range(m):
                nw_grid[:, i + 1, j + 1] = np.maximum(
                    np.maximum(nw_grid[:, i + 1, j], nw_grid[:, i, j + 1]),
                    nw_grid[:, i, j] + nw_scores[:, i, j])

        return nw_grid[:, -1, -1]