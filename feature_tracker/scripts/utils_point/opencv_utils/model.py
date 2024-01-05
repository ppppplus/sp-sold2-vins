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
  
class ORBPointMatchModel(BaseMatchModel):
    def _init(self, params=None):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match(self, data):
        desc1 = data["descriptors0"]
        desc2 = data["descriptors1"]
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        matches = self.bf.match(desc1.T, desc2.T)
        min_distance = matches[0].distance
        max_distance = matches[0].distance
        for x in matches:
            if x.distance < min_distance:
                min_distance = x.distance
            if x.distance > max_distance:
                max_distance = x.distance
                
        '''
            当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
            但有时候最小距离会非常小, 所以设置一个经验值30作为下限。
        '''
        good_match = []
        for x in matches:
            if x.distance <= max(2 * min_distance, 30):
                good_match.append([x.queryIdx, x.trainIdx, x.distance])
        return np.array(good_match).T

