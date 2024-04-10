import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
import yaml
from time import time
from . import common
from .patchnet import *
from utils.base_model import BaseExtractModel, BaseMatchModel
import torchvision.transforms as tvf

RGB_mean = [0.485, 0.456, 0.406]
RGB_std  = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=RGB_mean, std=RGB_std)])



def load_network(model_fn, cuda): 
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    if cuda: net = net.cuda()
    return net.eval()

class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        torch.nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]

class R2D2PointExtractModel(BaseExtractModel):
    def _init(self, params):
        self.params = params["r2d2"]
        self.H = params["H"]
        self.W = params["W"]
        self.border_remove = self.params["border_remove"]
        
        self.net = load_network(self.params["weights_path"], self.params["cuda"])
        self.detector = NonMaxSuppression(
            rel_thr = self.params["reliability_thr"],
            rep_thr = self.params["repeatability_thr"])
        
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
        # if img.ndim != 2:
        #     grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # else:
        #     grayim = img
        rgb_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = norm_RGB(rgb_image)[None] 
        # img = torch.from_numpy(img.astype(np.float32))[None]
        if self.params["cuda"]: img = img.cuda()
        # Image is resized via opencv.
        # interp = cv2.INTER_AREA
        # grayim = cv2.resize(grayim, (self.params['W'], self.params['H']), interpolation=interp)
        # grayim = (grayim.astype('float32') / 255.)
        return img, True

    def extract(self, img):
        # input img and output feature points   
        # This class runs the SuperPoint network and processes its outputs.
        grayim, status = self.process_image(img)
        if status is False:
            print("Load image error, Please check image_info topic")
            return
        
        # Get points and descriptors.
        with torch.no_grad():
            res = self.net(imgs=[grayim])
        # get output and reliability map
        descriptors = res['descriptors'][0]
        reliability = res['reliability'][0]
        repeatability = res['repeatability'][0]

        # normalize the reliability for nms
        # extract maxima and descs
        y,x = self.detector(**res) # nms
        c = reliability[0,0,y,x]
        q = repeatability[0,0,y,x]
        s = c*q
        d = descriptors[0,:,y,x]
        pts = torch.cat([x.unsqueeze(0),y.unsqueeze(0),s.unsqueeze(0)]).cpu().numpy()

        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (self.W-bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (self.H-bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        desc = d[:, ~toremove].cpu().numpy()
        # print("superpoint run time:%f", end_time-start_time)
        # feature_points = pts[0:2].T.astype('int32')
        return pts, desc # [3,num_points], [128,num_points]
