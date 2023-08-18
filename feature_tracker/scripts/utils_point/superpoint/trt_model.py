import os
import cv2
import numpy as np
import tensorrt as trt
import torch
import utils.common as common
from utils.base_model import BaseExtractModel
from time import time

TRT_LOGGER = trt.Logger()

class TrtSuperpointPointExtractModel(BaseExtractModel):
    def _init(self, params):
        self.params = params
        # self.fe = SuperPointFrontend(weights_path=self.params["weights_path"],
        #                         nms_dist=self.params["nms_dist"],
        #                         conf_thresh=self.params["conf_thresh"],
        #                         nn_thresh=self.params["nn_thresh"],
        #                         cuda=self.params["cuda"])  
        self.cell = 8
        self.border_remove = 4
        self.conf_thresh=self.params["conf_thresh"]
        self.nms_dist = self.params["nms_dist"]
        self.engine = self.get_engine(self.params["engine_file_path"])
        self.context = self.engine.create_execution_context()
        

    def extract(self, img):
        grayim, status = self.process_image(img)
        if status is False:
            print("Load image error, Please check image_info topic")
            return
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
        # Do inference
        # print("Running inference on image {}...".format(img_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = grayim
        # start_time = time()
        trt_outputs = common.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        # end_time = time()
        # print(trt_outputs[0].shape)
        # print(trt_outputs[1].shape)
        # line_segments = trt_outputs[0].reshape((2,2,...))
        # line_desc = trt.outputs[1].reshape((128,...))
        # print(line_segments.shape, line_desc.shape)
        semi = trt_outputs[0].reshape((65,self.H//self.cell,self.W//self.cell))
        coarse_desc = trt_outputs[1].reshape((1,256,self.H//self.cell,self.W//self.cell))
        pts, desc = self.process_output(semi, coarse_desc)
        return pts, desc

    def process_output(self, semi, coarse_desc):
        semi = semi.squeeze()
        dense = np.exp(semi) # Softmax.
        dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(self.H / self.cell)
        Wc = int(self.W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
        pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, self.H, self.W, dist_thresh=self.nms_dist) # Apply NMS.
        inds = np.argsort(pts[2,:])
        pts = pts[:,inds[::-1]] # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (self.W-bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (self.H-bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(self.W)/2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(self.H)/2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(torch.from_numpy(coarse_desc).cuda(), samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc
    
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
        self.H, self.W = img.shape[0], img.shape[1]
        # Image is resized via opencv.
        # interp = cv2.INTER_AREA
        # grayim = cv2.resize(grayim, (self.params['W'], self.params['H']), interpolation=interp)
        grayim = (grayim.astype('float32') / 255.)[None, None]
        return grayim, True
    
    def get_engine(self, engine_file_path):
        """Attempts to load a serialized engine if available."""
        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            raise ValueError("Engine file {} does not exist!".format(engine_file_path))
    def nms_fast(self, in_corners, H, W, dist_thresh):
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