#!/usr/bin/env python

import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
import yaml
from utils.base_model import BaseExtractModel, BaseMatchModel

class ORBPointExtractModel(BaseExtractModel):
  def _init(self, params=None):
    self.orb = cv2.ORB_create()

  def extract(self, img):
    # input img and output feature points&descriptors

    if img is None:
        print("Load image error, Please check image_info topic")
        return
    # Get points and descriptors.
    kpts = self.orb.detect(img)
    kpts, desc = self.orb.compute(img, kpts)
    pts = cv2.KeyPoint_convert(kpts)
    return pts.T, desc.T # [2,num_points], [32, num_points]
    # return pts.T, desc
  
class ORBPointMatchModel(BaseMatchModel):
    def _init(self, params=None):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, data):
        desc1 = data["descriptors0"]
        desc2 = data["descriptors1"]
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        
        pairs_of_matches = self.bf.knnMatch(desc1.T, desc2.T, k=2)
        matches = [x[0] for x in pairs_of_matches
                if len(x) > 1 and x[0].distance < 0.7 * x[1].distance]

        good_match = []
        for x in matches:
            good_match.append([x.queryIdx, x.trainIdx, x.distance])
        return np.array(good_match).T

