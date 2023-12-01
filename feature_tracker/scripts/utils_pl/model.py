import torch
import cv2
import torch.nn as nn
import torch.nn.init as init
from utils.base_model import BaseExtractModel, BaseMatchModel
from .line_detection import LineSegmentDetectionModule
from .nets.backbone import HourglassBackbone, SuperpointBackbone
from .nets.junction_decoder import SuperpointDecoder
from .nets.heatmap_decoder import PixelShuffleDecoder
from .nets.descriptor_decoder import SuperpointDescriptor
# from time import time 
import time
import numpy as np
from .metrics import super_nms, line_map_to_segments
from pyinstrument import Profiler

class SPSOLD2Net(nn.Module):
    """ Full network for SOLD². """
    def __init__(self, model_cfg):
        super(SPSOLD2Net, self).__init__()
        self.name = model_cfg["model_name"]
        self.cfg = model_cfg

        # List supported network options
        self.supported_backbone = ["superpoint"]
        self.backbone_net, self.feat_channel = self.get_backbone()

        # List supported junction decoder options
        self.supported_junction_decoder = ["superpoint_decoder"]
        self.junction_decoder = self.get_junction_decoder()

        # List supported heatmap decoder options
        self.supported_heatmap_decoder = ["pixel_shuffle",
                                          "pixel_shuffle_single"]
        self.heatmap_decoder = self.get_heatmap_decoder()

        # List supported descriptor decoder options
        if "descriptor_decoder" in self.cfg:
            self.supported_descriptor_decoder = ["superpoint_descriptor"]
            self.descriptor_decoder = self.get_descriptor_decoder()

        # Initialize the model weights
        self.apply(weight_init)

    def forward(self, input_images):
        # The backbone
        features = self.backbone_net(input_images)

        # junction decoder
        junctions = self.junction_decoder(features)

        # heatmap decoder
        heatmaps = self.heatmap_decoder(features)

        # descriptor decoder
        descriptors = self.descriptor_decoder(features)

        outputs = {"junctions": junctions, "heatmap": heatmaps, "descriptors": descriptors}

        return outputs

    def get_backbone(self):
        """ Retrieve the backbone encoder network. """
        if not self.cfg["backbone"] in self.supported_backbone:
            raise ValueError(
                "[Error] The backbone selection is not supported.")

        if self.cfg["backbone"] == "superpoint":
            backbone_cfg = self.cfg["backbone_cfg"]
            backbone = SuperpointBackbone()
            feat_channel = 128

        else:
            raise ValueError(
                "[Error] The backbone selection is not supported.")

        return backbone, feat_channel

    def get_junction_decoder(self):
        """ Get the junction decoder. """
        if (not self.cfg["junction_decoder"]
            in self.supported_junction_decoder):
            raise ValueError(
                "[Error] The junction decoder selection is not supported.")

        # superpoint decoder
        if self.cfg["junction_decoder"] == "superpoint_decoder":
            decoder = SuperpointDecoder(self.feat_channel,
                                        self.cfg["backbone"])
        else:
            raise ValueError(
                "[Error] The junction decoder selection is not supported.")

        return decoder

    def get_heatmap_decoder(self):
        """ Get the heatmap decoder. """
        if not self.cfg["heatmap_decoder"] in self.supported_heatmap_decoder:
            raise ValueError(
                "[Error] The heatmap decoder selection is not supported.")

        # Pixel_shuffle decoder
        if self.cfg["heatmap_decoder"] == "pixel_shuffle":
            if self.cfg["backbone"] == "lcnn":
                decoder = PixelShuffleDecoder(self.feat_channel,
                                              num_upsample=2)
            elif self.cfg["backbone"] == "superpoint":
                decoder = PixelShuffleDecoder(self.feat_channel,
                                              num_upsample=3)
            else:
                raise ValueError("[Error] Unknown backbone option.")
        # Pixel_shuffle decoder with single channel output
        elif self.cfg["heatmap_decoder"] == "pixel_shuffle_single":
            if self.cfg["backbone"] == "lcnn":
                decoder = PixelShuffleDecoder(
                    self.feat_channel, num_upsample=2, output_channel=1)
            elif self.cfg["backbone"] == "superpoint":
                decoder = PixelShuffleDecoder(
                    self.feat_channel, num_upsample=3, output_channel=1)
            else:
                raise ValueError("[Error] Unknown backbone option.")
        else:
            raise ValueError(
                "[Error] The heatmap decoder selection is not supported.")

        return decoder

    def get_descriptor_decoder(self):
        """ Get the descriptor decoder. """
        if (not self.cfg["descriptor_decoder"]
            in self.supported_descriptor_decoder):
            raise ValueError(
                "[Error] The descriptor decoder selection is not supported.")

        # SuperPoint descriptor
        if self.cfg["descriptor_decoder"] == "superpoint_descriptor":
            decoder = SuperpointDescriptor(self.feat_channel)
        else:
            raise ValueError(
                "[Error] The descriptor decoder selection is not supported.")

        return decoder
        
class SPSOLD2ExtractModel(BaseExtractModel):
    def _init(self, params):
        self.params = params
        self.grid_size = params["grid_size"]
        self.pad_size = params["pad_size"]
        self.H = params["H"]
        self.W = params["W"]
        self.Hc = self.H//self.grid_size
        self.Wc = self.W//self.grid_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SPSOLD2Net(params)
        self.model.eval()
        self.model.to(self.device)

        model_state_dict = torch.load(params["ckpt_path"])
        self.model.load_state_dict(model_state_dict["model_state_dict"], strict=False)

        self.conf_thresh = params["conf_thresh"]
        self.nms_dist = params["nms_dist"]
        self.border_remove = params["border_remove"]
        self.num_samples = params["num_samples"]
        self.min_dist_pts = params["min_dist_pts"]
        self.line_score = params["line_score"]
        self.sampling = params["sampling"]
        self.detection_thresh = params["detection_thresh"]
        self.topk = params["topk"]
        
        line_detector_cfg = params["line_detector_cfg"]
        self.line_detector = LineSegmentDetectionModule(**line_detector_cfg)
        

    def process_image(self, img):
        """ convert image to grayscale and resize to img_size.
        Inputs
        impath: Path to input image.
        img_size: (W, H) tuple specifying resize size.
        Returns
        grayim: float32 numpy array sized H x W with values in range [0, 1].
        """
        if img is None:
            return (None, False)
        if img.ndim != 2:
            grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            grayim = img
        
        # Image is resized via opencv.
        # interp = cv2.INTER_AREA
        # grayim = cv2.resize(grayim, (self.params['W'], self.params['H']), interpolation=interp)
        grayim = (grayim.astype('float32') / 255.)
        torch_img = torch.tensor(grayim, dtype=torch.float).to(self.device)[None, None]
        return torch_img, True

    def extract(self, img):
        # input img and output feature points   
        # This class runs the SuperPoint network and processes its outputs.
        #########################################################
        torch_img, status = self.process_image(img)
        if status is False:
            print("Load image error, Please check image_info topic")
            return
        
        # Get points and descriptors.
        torch.cuda.synchronize()
        infstime = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(torch_img)
        torch.cuda.synchronize()
        infetime = time.perf_counter()
        print("inference time: ", infetime-infstime)
        junctions = outputs["junctions"]
        heatmap = outputs["heatmap"]
        coarse_desc  = outputs["descriptors"]
        ########################################################
        torch.cuda.synchronize()
        pstime = time.perf_counter()
        pts, junc_np = self.convert_junc_predictions(junctions)
        pts, pts_desc = self.reprocess_pts_np(pts, coarse_desc) # [3, n_pts]; [128, n_pts]
        torch.cuda.synchronize()
        petime = time.perf_counter()
        print("pt extract time: ", petime-pstime)
        torch.cuda.synchronize()
        lstime = time.perf_counter()
        lines, lines_desc, valid_points = self.reprocess_lines_np(junc_np, heatmap, coarse_desc)
        torch.cuda.synchronize()
        letime = time.perf_counter()
        print("line extract time: ",letime-lstime)
        
        # print("superpoint run time:%f", end_time-start_time)
        # feature_points = pts[0:2].T.astype('int32')
        ############################################################
        # return pts, pts_desc, lines, lines_desc, valid_points
    
    def reprocess_pts_np(self, pts, coarse_desc):
        profiler = Profiler()
        profiler.start()
        ############################# nms #############################
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (self.W-bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (self.H-bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = torch.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = pts[:2, :]
            samp_pts = torch.tensor(samp_pts, dtype=torch.float32).to(self.device)
            samp_pts[0, :] = (samp_pts[0, :] / (float(self.W)/2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(self.H)/2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            desc = nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.reshape(D, -1)
            desc /= torch.norm(desc, dim=0)
        profiler.stop()
        profiler.print()
        return pts, desc
    
    def reprocess_pts(self, junctions, coarse_desc):
        # reprocess net outputs and get pts&desc
        ### junctions: 1,65,Hc,Wc
        ### heatmap: 1,2,H,W
        ### coarse_desc: 1,128,Hc,Wc
        # time1 = time()
        semi = junctions.squeeze()
        dense = torch.exp(semi) # Softmax.
        dense = dense / (torch.sum(dense, axis=0)+.00001) # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        
        nodust = nodust.permute(1, 2, 0)
        heatmap = torch.reshape(nodust, [self.Hc, self.Wc, self.grid_size, self.grid_size])
        heatmap = heatmap.permute(0, 2, 1, 3)
        heatmap = torch.reshape(heatmap, [self.Hc*self.grid_size, self.Wc*self.grid_size])
        xs, ys = torch.where(heatmap >= self.conf_thresh) # Confidence threshold.
        pts = torch.zeros((3, len(xs))).to(self.device) # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        # pts[2, :] = heatmap[xs, ys]
        # print("time1: ", time()-time1)
        pts, _ = self.nms_fast_torch(pts, self.H, self.W, dist_thresh=self.nms_dist) # Apply NMS.
        inds = torch.argsort(pts[2,:])
        inds = torch.flip(inds, dims=[0])
        pts = pts[:,inds] # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = torch.logical_or(pts[0, :] < bord, pts[0, :] >= (self.W-bord))
        toremoveH = torch.logical_or(pts[1, :] < bord, pts[1, :] >= (self.H-bord))
        toremove = torch.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        # print("time2: ", time()-time1)
        if pts.shape[1] == 0:
            desc = torch.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = pts[:2, :]
            samp_pts[0, :] = (samp_pts[0, :] / (float(self.W)/2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(self.H)/2.)) - 1.
            samp_pts = samp_pts.transpose(     0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.cuda()
            desc = nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.reshape(D, -1)
            desc /= torch.norm(desc, dim=0)
            # desc = desc.reshape(D, -1)
            # desc = torch.norm(desc)[None]
        # print("time3: ", time()-time1)
        return pts, desc
    
    def nms_fast_np(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
        3xN [x_i,y_i,conf_i]^T
    
        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.
    
        Grid Value Legend:
        -1 : Kept.
        0 : Empty or suppressed.
        1 : To be processed (converted to either kept or supressed).
    
        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.
    
        Inputs
        in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
        H - Image height.
        W - Image width.
        dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
        nmsed_corners - 3xN numpy matrix with surviving corners.
        nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int) # Track NMS data.
        inds = np.zeros((H, W)).astype(int) # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2,:])
        corners = in_corners[:,inds1]
        rcorners = corners[:2,:].round().astype(int) # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1,i], rcorners[0,i]] = 1
            inds[rcorners[1,i], rcorners[0,i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0]+pad, rc[1]+pad)
            if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
                grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid==-1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def nms_fast_torch(self, in_corners, H, W, dist_thresh):
        grid = torch.zeros((H, W), dtype=int).to(self.device) # Track NMS data.
        inds = torch.zeros((H, W), dtype=int).to(self.device) # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = torch.argsort(-in_corners[2,:])
        corners = in_corners[:,inds1]
        rcorners = corners[:2,:].round().int() # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return torch.zeros((3,0)).int(), torch.zeros(0).int()
        if rcorners.shape[1] == 1:
            out = torch.vstack((rcorners, in_corners[2])).reshape(3,1)
            return out, torch.zeros((1), dtype=int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1,i], rcorners[0,i]] = 1
            inds[rcorners[1,i], rcorners[0,i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = torch.nn.functional.pad(grid, (pad,pad,pad,pad), mode='constant', value=0)
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0]+pad, rc[1]+pad)
            if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
                grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = torch.where(grid==-1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = torch.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds
    
    def reprocess_lines_np(self, junc_np, heatmap, coarse_desc):
        # profiler = Profiler()
        # profiler.start()
        junctions = np.where(junc_np.squeeze())
        junctions = np.concatenate(
            [junctions[0][..., None], junctions[1][..., None]], axis=-1)
        if heatmap.shape[1] == 2:
            # Convert to single channel directly from here
            heatmap = nn.functional.softmax(heatmap, dim=1)[:, 1:, :, :]
        else:
            heatmap = torch.sigmoid(heatmap)
        # heatmap = heatmap.cpu().numpy().transpose(0, 2, 3, 1)[0, :, :, 0]
        heatmap = heatmap.squeeze()
        # Run the line detector.
        line_map, junctions, heatmap = self.line_detector.detect(
            junctions, heatmap, device=self.device) 
        # time2 = time()
        # print("time1:", time11-time1, time2-time11) # 最耗时：0.5s
        # heatmap = heatmap.cpu().numpy()
        # if isinstance(line_map, torch.Tensor):
        #     line_map = line_map.cpu().numpy()
        # if isinstance(junctions, torch.Tensor):
        #     junctions = junctions.cpu().numpy()
        line_segments = self.line_map_to_segments(junctions, line_map)  # 第二耗时 0.23s
        line_segments = torch.tensor(line_segments, dtype=torch.float32).to(self.device)
        line_points, valid_points = self.sample_line_points(line_segments)  

        line_points = line_points.reshape(-1, 2)
        # Extract the descriptors for each point
        grid = self.keypoints_to_grid(line_points)
        line_desc = nn.functional.normalize(nn.functional.grid_sample(coarse_desc, grid)[0, :, :, 0], dim=0)
        line_desc = line_desc.reshape((-1, len(line_segments), self.num_samples)) # reshape为每个线段对应的desc,128*num_lines*num_samples
        # profiler.stop()

        # profiler.print()
        return line_segments.cpu().numpy(), line_desc, valid_points.cpu().numpy()

    def remove_borders(self, keypoints, scores, border: int):
        """ Removes keypoints too close to the border """
        mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (self.H - border))
        mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (self.W - border))
        mask = mask_h & mask_w
        return keypoints[mask], scores[mask]

    def top_k_keypoints(self, keypoints, scores, k: int):
        if k >= len(keypoints):
            return keypoints, scores
        scores, indices = torch.topk(scores, k, dim=0)
        return keypoints[indices], scores

    def sample_descriptors(self, keypoints, descriptors, s: int = 8):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape
        keypoints = keypoints - s / 2 + 0.5
        keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                                ).to(keypoints)[None]
        keypoints = keypoints*2 - 1  # normalize to (-1, 1)
        args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors


    def reprocess_pts_bp(self, junctions, coarse_desc):
        # reprocess net outputs and get pts&desc
        ### junctions: 1,65,Hc,Wc
        ### heatmap: 1,2,H,W
        ### coarse_desc: 1,128,Hc,Wc
        
        # semi = junctions.squeeze()
        # profiler = Profiler()
        # profiler.start()
        # torch.cuda.synchronize()
        a = time.perf_counter()
        ########################## GPU #################################
        scores = torch.nn.functional.softmax(junctions, dim=1)[:, :-1, ...]
        # Reshape to get full resolution heatmap.
        
        scores = scores.permute(0, 2, 3, 1)    # Hc*Wc*grid_size^2
        
        heatmap = torch.reshape(scores, [1, self.Hc, self.Wc, self.grid_size, self.grid_size])
        heatmap = heatmap.permute(0, 1, 3, 2, 4)
        heatmap = torch.reshape(heatmap, [1, self.Hc*self.grid_size, self.Wc*self.grid_size])
        heatmap = self.simple_nms(heatmap, self.nms_dist)
        # heatmap = heatmap.cpu().numpy()
        # profiler.stop()
        # profiler.print()
        #####################################################################
        ######################### CPU ####################################
        # profiler = Profiler()
        # profiler.start()
        # semi = junctions.data.cpu().numpy().squeeze()
        # # --- Process points.
        # dense = np.exp(semi) # Softmax.
        # dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
        # # Remove dustbin.
        # nodust = dense[:-1, :, :]
        # profiler.stop()
        # profiler.print()
        # # Reshape to get full resolution heatmap.
        # # Hc = int(self.H / self.cell)
        # # Wc = int(self.W / self.cell)
        # nodust = nodust.transpose(1, 2, 0)
        # heatmap = np.reshape(nodust, [self.Hc, self.Wc, self.grid_size, self.grid_size])
        # heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        # heatmap = np.reshape(heatmap, [self.Hc*self.grid_size, self.Wc*self.grid_size])
        # heatmap = self.simple_nms_np(heatmap)
        #############################################################
        # Extract keypoints
       
        
        keypoints = [
            torch.nonzero(s > self.conf_thresh)
            for s in heatmap]

        # torch.cuda.synchronize()
        b = time.perf_counter()
        print(b-a)
        # bindex = heatmap>self.conf_thresh
        # bindex = bindex.cpu().numpy()
        # keypoint = []
        # for i in range(bindex.shape[1]):
        #     for j in range(bindex.shape[2]):
        #         if bindex[0,i,j] > self.conf_thresh:
        #             keypoint.append([i,j])
        # zeros = torch.zeros(bindex.shape).to(self.device)
        # ones=torch.ones(x.shape)
        # keypoints = torch.where(bindex,zeros,heatmap)
        # bindex = bindex.squeeze()
        # # a = torch.nonzero(bindex)
        # bindex = bindex.cpu().numpy()
        # a = np.where(bindex>0)
        
        # keypoints = torch.where(heatmap>self.conf_thresh)
        # keypoints = torch.tensor(keypoints).to(self.device)
        scores = [s[tuple(k.t())] for s, k in zip(heatmap, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            self.remove_borders(k, s, self.border_remove)
            for k, s in zip(keypoints, scores)])) 

        # Keep the k keypoints with highest score
        # if self.config['max_keypoints'] >= 0:
        #     keypoints, scores = list(zip(*[
        #         top_k_keypoints(k, s, self.config['max_keypoints'])
        #         for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        pts = [torch.flip(k, [1]).float() for k in keypoints]

        # --- Process descriptor.
        descriptors = torch.nn.functional.normalize(coarse_desc, p=2, dim=1)

        # Extract descriptors
        descriptors = [self.sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]
        
        # D = coarse_desc.shape[1]
        # if len(pts) == 0:
        #     desc = np.zeros((D, 0))
        # else:
        #     # Interpolate into descriptor map using 2D point locations.
        #     samp_pts = torch.from_numpy(pts[:2, :].copy())
        #     samp_pts[0, :] = (samp_pts[0, :] / (float(self.W)/2.)) - 1.
        #     samp_pts[1, :] = (samp_pts[1, :] / (float(self.H)/2.)) - 1.
        #     samp_pts = samp_pts.transpose(0, 1).contiguous()
        #     samp_pts = samp_pts.view(1, 1, -1, 2)
        #     samp_pts = samp_pts.float()
        #     samp_pts = samp_pts.cuda()
        # desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
        # desc = desc.data.cpu().numpy().reshape(D, -1)
        # desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

       
        return pts, descriptors, heatmap
    
    def reprocess_lines(self, junctions, heatmap, coarse_desc):
        # reprocess net outputs and get lines&desc
        # junc_pred = self.convert_junc_predictions(junctions)
        junctions = torch.where(junctions.squeeze()) 
    
        junctions = torch.cat([junctions[0][..., None],
                                    junctions[1][..., None]], axis=-1)

        if heatmap.shape[1] == 2:
            # Convert to single channel directly from here
            heatmap = nn.functional.softmax(heatmap, dim=1)[:, 1:, :, :].permute(0, 2, 3, 1)
        else:
            heatmap = torch.sigmoid(heatmap).permute(0, 2, 3, 1)
        heatmap = heatmap[0, :, :, 0]

        # Run the line detector.
        line_map, junctions, heatmap = self.line_detector.detect(
            junctions, heatmap, device=self.device)
        # if isinstance(line_map, torch.Tensor):
        #     line_map = line_map.cpu().numpy()
        # if isinstance(junctions, torch.Tensor):
        #     junctions = junctions.cpu().numpy()
        # outputs["heatmap"] = heatmap
        # outputs["junctions"] = junctions

        # If it's a line map with multiple detect_thresh and inlier_thresh

        line_segments = self.line_map_to_segments(junctions, line_map)


        # end_time = time.time()
        # 参照line_matching中的函数做描述子和线之间的对应
        line_segments = torch.tensor(line_segments, dtype=torch.float32).to(self.device)
        line_points, valid_points = self.sample_line_points(line_segments)  
        
        line_points = line_points.reshape(-1, 2)
        # Extract the descriptors for each point
        grid = self.keypoints_to_grid(line_points)
        line_desc = nn.functional.normalize(nn.functional.grid_sample(coarse_desc, grid)[0, :, :, 0], dim=0)
        line_desc = line_desc.reshape((-1, len(line_segments), self.num_samples)) # reshape为每个线段对应的desc,128*num_lines*num_samples

        return line_segments, line_desc, valid_points
        # if profile:
        #     outputs["time"] = end_time - start_time

    def keypoints_to_grid(self, keypoints):
        n_points = keypoints.size()[-2]
        grid_points = keypoints.float() * 2. / torch.tensor(
            [self.H, self.W], dtype=torch.float, device=self.device) - 1.
        grid_points = grid_points[..., [1, 0]].view(-1, n_points, 1, 2)
        return grid_points
    
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
        line_lengths = torch.norm(line_seg[:, 0] - line_seg[:, 1], dim=1)

        # Sample the points separated by at least min_dist_pts along each line
        # The number of samples depends on the length of the line
        num_samples_lst = torch.clip(line_lengths // self.min_dist_pts,
                                  2, self.num_samples)
        line_points = torch.empty((num_lines, self.num_samples, 2), dtype=float).to(self.device)
        valid_points = torch.empty((num_lines, self.num_samples), dtype=bool).to(self.device)
        for n in np.arange(2, self.num_samples + 1):
            # Consider all lines where we can fit up to n points
            cur_mask = num_samples_lst == n
            cur_line_seg = line_seg[cur_mask]
            line_points_x = np.linspace(cur_line_seg[:, 0, 0].cpu().numpy(),
                                        cur_line_seg[:, 1, 0].cpu().numpy(),
                                        n, axis=-1)
            line_points_y = np.linspace(cur_line_seg[:, 0, 1].cpu().numpy(),
                                        cur_line_seg[:, 1, 1].cpu().numpy(),
                                        n, axis=-1)
            cur_line_points = np.stack([line_points_x, line_points_y], axis=-1)
            cur_line_points = torch.from_numpy(cur_line_points).to(self.device)
            # Pad
            cur_num_lines = len(cur_line_seg)
            cur_valid_points = torch.ones((cur_num_lines, self.num_samples),
                                       dtype=bool).to(self.device)
            cur_valid_points[:, n:] = False
            cur_line_points = torch.cat([
                cur_line_points,
                torch.zeros((cur_num_lines, self.num_samples - n, 2), dtype=float).to(self.device)],
                axis=1)

            line_points[cur_mask] = cur_line_points
            valid_points[cur_mask] = cur_valid_points

        return line_points, valid_points
    
    # def line_map_to_segments(self, junctions, line_map):
    #     profiler = Profiler()
    #     profiler.start()
    #     """ Convert a line map to a Nx2x2 list of segments. """ 
    #     line_map_tmp = line_map.clone()

    #     output_segments = torch.zeros([0, 2, 2]).to(self.device)
    #     for idx in range(junctions.shape[0]):
    #         # if no connectivity, just skip it
    #         if line_map_tmp[idx, :].sum() == 0:
    #             continue
    #         # Record the line segment
    #         else:
    #             for idx2 in torch.where(line_map_tmp[idx, :] == 1)[0]:
    #                 p1 = junctions[idx, :]  # HW format
    #                 p2 = junctions[idx2, :]
    #                 single_seg = torch.cat([p1[None, ...], p2[None, ...]],
    #                                             axis=0)
    #                 output_segments = torch.cat(
    #                     (output_segments, single_seg[None, ...]), axis=0)
                    
    #                 # Update line_map
    #                 line_map_tmp[idx, idx2] = 0
    #                 line_map_tmp[idx2, idx] = 0
    #     profiler.stop()
    #     profiler.print()
    #     return output_segments

    def line_map_to_segments(self, junctions, line_map):
        # profiler = Profiler()
        # profiler.start()
        """ Convert a line map to a Nx2x2 list of segments. """
        junctions = junctions.cpu().numpy() 
        line_map_tmp = line_map.cpu().numpy()

        output_segments = []
        for idx in range(junctions.shape[0]):
            # if no connectivity, just skip it
            if np.sum(line_map_tmp[idx, :]) == 0:
                continue
            # Record the line segment
            else:
                for idx2 in np.where(line_map_tmp[idx, :] == 1)[0]:
                    p1 = junctions[idx, :]  # HW format
                    p2 = junctions[idx2, :]
                    single_seg = np.array([p1,p2])
                    output_segments.append(single_seg)
                    
                    # Update line_map
                    line_map_tmp[idx, idx2] = 0
                    line_map_tmp[idx2, idx] = 0
        output_segments = np.array(output_segments)
        # profiler.stop()
        # profiler.print()
        return output_segments

    def convert_junc_predictions(self, predictions):
        # profiler = Profiler()
        # profiler.start()
        """ Convert torch predictions to numpy arrays for evaluation. """
        #########################simple_nms##########################
        junc_prob = nn.functional.softmax(predictions, dim=1)
        junc_pred = junc_prob[:, :-1, :, :]
        junc_pred = nn.functional.pixel_shuffle(
            junc_pred, self.grid_size).permute(0, 2, 3, 1) # [1,H,W,1]
        junc_pred = junc_pred[..., 0]
        junc_pred_nms = self.simple_nms(junc_pred, self.grid_size)
        coord = torch.where(junc_pred_nms >= self.conf_thresh) # HW format
        points = torch.cat((coord[0][..., None], coord[1][..., None]),
                                axis=1) # HW format

        # Get the probability score
        junc_pred_nms = junc_pred_nms.squeeze()
        prob_score = junc_pred_nms[points[:, 0], points[:, 1]]

        # Perform super nms
        # Modify the in_points to xy format (instead of HW format)
        in_points = torch.cat((coord[1][..., None], coord[0][..., None],
                                    prob_score[..., None]), axis=1).T
        return in_points.cpu().numpy(), junc_pred_nms.cpu().numpy()
        #########################simple_nms##########################
        # Convert to probability outputs first
        # junc_prob = nn.functional.softmax(predictions, dim=1)
        # # junc_np = predictions.cpu().numpy()    # 1*65*Hc*Wc
        # # junc_exp = np.exp(junc_no_dust)
        # junc_pred = junc_prob[:, :-1, :, :]
        # # junc_pred_np = nn.functional.pixel_shuffle(
        # #     junc_pred, self.grid_size).transpose(0, 2, 3, 1)
        # junc_pred = nn.functional.pixel_shuffle(
        #     junc_pred, self.grid_size).permute(0, 2, 3, 1)
        # junc_pred_np_nms = self.simple_nms(junc_pred, self.grid_size)
        # pts, junc_np_nms = self.softmax_nms(junc_np, self.pad_size, self.detection_thresh, self.topk)
        # profiler.stop()
        # profiler.print()
        # return junc_pred_np_nms
    
    def softmax_nms(self, prob_map, pad_size, prob_thresh, top_k):
        """
        softmax+nms, only do softmax on filtered points
        """
        pts = [] 
        junc_np_nms = np.zeros([self.H, self.W])
        for i in range(prob_map.shape[0]):
            prob_pred_grids = prob_map[i, ...]
            prob_pred_grids = prob_pred_grids.reshape((65, -1)).transpose(1, 0)   # (Hc*Wc)*65
            for j in range(prob_pred_grids.shape[0]):
                prob_pred_grid = prob_pred_grids[j]
                dust = prob_pred_grid[-1]
                no_dust = prob_pred_grid[:-1]
                prob_grid_sorted_index = np.argsort(-no_dust)
                prob_grid_sorted = no_dust[prob_grid_sorted_index]
                prob_grid_max_index = prob_grid_sorted_index[0]
                prob_grid_max = prob_grid_sorted[0]
                prob_grid_exp_index = (prob_grid_sorted>0)&(prob_grid_sorted>0.1*prob_grid_max)
                prob_grid_exp = prob_grid_sorted[prob_grid_exp_index]
                if len(prob_grid_exp) != 0:
                    prob_grid_exp = np.exp(prob_grid_exp)
                    prob_grid_softmax = prob_grid_exp[0]/(np.sum(prob_grid_exp)+np.exp(dust))
                    prob_grid_nms_x = j//self.Wc*self.grid_size+(prob_grid_max_index//self.grid_size)
                    prob_grid_nms_y = j%self.Wc*self.grid_size+(prob_grid_max_index%self.grid_size)
                    pts.append([prob_grid_nms_x, prob_grid_nms_y, prob_grid_softmax])
                    junc_np_nms[prob_grid_nms_x, prob_grid_nms_y] = prob_grid_softmax
        pts = np.array(pts)
        inds = np.argsort(-pts[2,:])
        pts = pts[:,inds] # Sort by confidence.
        return pts, junc_np_nms
            
    def super_nms(self, prob_predictions, dist_thresh, prob_thresh=0.01, top_k=0):
        """ Non-maximum suppression adapted from SuperPoint. """
        # Iterate through batch dimension
        im_h = prob_predictions.shape[1]
        im_w = prob_predictions.shape[2]
        output_lst = []
        for i in range(prob_predictions.shape[0]):
            # print(i)
            prob_pred = prob_predictions[i, ...]
            # Filter the points using prob_thresh
            coord = np.where(prob_pred >= prob_thresh) # HW format
            points = np.concatenate((coord[0][..., None], coord[1][..., None]),
                                    axis=1) # HW format

            # Get the probability score
            prob_score = prob_pred[points[:, 0], points[:, 1]]

            # Perform super nms
            # Modify the in_points to xy format (instead of HW format)
            in_points = np.concatenate((coord[1][..., None], coord[0][..., None],
                                        prob_score), axis=1).T
            keep_points_, keep_inds = self.nms_fast(in_points, im_h, im_w, dist_thresh)
            # Remember to flip outputs back to HW format
            keep_points = np.round(np.flip(keep_points_[:2, :], axis=0).T)
            keep_score = keep_points_[-1, :].T

            # Whether we only keep the topk value
            if (top_k > 0) or (top_k is None):
                k = min([keep_points.shape[0], top_k])
                keep_points = keep_points[:k, :]
                keep_score = keep_score[:k]

            # Re-compose the probability map
            output_map = np.zeros([im_h, im_w])
            output_map[keep_points[:, 0].astype(np.int32),
                    keep_points[:, 1].astype(np.int32)] = keep_score.squeeze()

            output_lst.append(output_map[None, ...])

        return keep_points_, np.concatenate(output_lst, axis=0)

    
    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
        3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1,
        rest are zeros. Iterate through all the 1's and convert them to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
        0 : Empty or suppressed.
        1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundary.

        Inputs
        in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
        H - Image height.
        W - Image width.
        dist_thresh - Distance to suppress, measured as an infinite distance.
        Returns
        nmsed_corners - 3xN numpy matrix with surviving corners.
        nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def simple_nms(self, heatmap, nms_radius: int):
        """ Fast Non-maximum suppression to remove nearby points """
        assert(nms_radius >= 0)
        zeros = torch.zeros_like(heatmap)
        max_mask = heatmap == max_pool(heatmap, nms_radius)
        for _ in range(1):
            supp_mask = max_pool(max_mask.float(), nms_radius) > 0
            supp_scores = torch.where(supp_mask, zeros, heatmap)
            new_max_mask = supp_scores == max_pool(supp_scores, nms_radius)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, heatmap, zeros)
    
    def simple_nms_np(self, heatmap, nms_radius: int):
        """ Fast Non-maximum suppression to remove nearby points """
        assert(nms_radius >= 0)
        # zeros = torch.zeros_like(heatmap)
        max_mask = heatmap == max_pool(heatmap, nms_radius)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float(), nms_radius) > 0
            supp_scores = torch.where(supp_mask, zeros, heatmap)
            new_max_mask = supp_scores == max_pool(supp_scores, nms_radius)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return np.where(max_mask, heatmap, 0)
