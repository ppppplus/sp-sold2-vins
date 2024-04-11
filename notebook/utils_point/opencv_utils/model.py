#!/usr/bin/env python

import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
import yaml
from pylab import *
from PCV.localdescriptors import harris
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
  
class FASTPointExtractModel(BaseExtractModel):
    def _init(self, params=None):
        self.fast = cv2.FastFeatureDetector_create()

    def process_image(self, img):
        if img is None:
            return (None, False)
        if img.ndim != 2:
            grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            grayim = img
        return grayim, True

    def extract(self, img):
        # input img and output feature points&descriptors

        grayim, status = self.process_image(img)
        if status is False:
            print("Load image error, Please check image_info topic")
            return
        # Get points and descriptors..
        kpts = self.fast.detect(grayim, None)
        kpts, desc = self.fast.detectAndCompute(img, None)
        pts = cv2.KeyPoint_convert(kpts)
        return pts.T, desc.T # [2,num_points], [32, num_points]
        # return pts.T, desc
    
class HarrisPointExtractModel(BaseExtractModel):
    def _init(self, params=None):
        self.wid = 5

    def process_image(self, img):
        if img is None:
            return (None, False)
        if img.ndim != 2:
            grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            grayim = img
        return grayim, True

    def extract(self, img):
        # input img and output feature points&descriptors
        # if img is None:
        #     print("Load image error, Please check image_info topic")
        #     return
        grayim, status = self.process_image(img)
        if status is False:
            print("Load image error, Please check image_info topic")
            return
        harrisim = harris.compute_harris_response(grayim, self.wid)
        filtered_coords = harris.get_harris_points(harrisim, self.wid+1)
        desc = harris.get_descriptors(grayim, filtered_coords, self.wid)
        # pts = cv2.KeyPoint_convert(filtered_coords)
        pts = np.array(filtered_coords)
        desc = np.array(desc)
        return pts.T, desc.T # [2,num_points], [121, num_points]
        # return pts.T, desc
    
class SIFTPointExtractModel(BaseExtractModel):
    def _init(self, params=None):
        self.sift = cv2.SIFT_create()

    def process_image(self, img):
        if img is None:
            return (None, False)
        if img.ndim != 2:
            grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            grayim = img
        return grayim, True

    def extract(self, img):
        # input img and output feature points&descriptors
        # if img is None:
        #     print("Load image error, Please check image_info topic")
        #     return
        if img is None:
            print("Load image error, Please check image_info topic")
            return
        # Get points and descriptors.
        kpts, desc = self.sift.detectAndCompute(img, None)
        pts = cv2.KeyPoint_convert(kpts)
        return pts.T, desc.T # [2,num_points], [121, num_points]
        # return pts.T, desc
    
class SURFPointExtractModel(BaseExtractModel):
    def _init(self, params=None):
        self.surf = cv2.SURF_create()

    def process_image(self, img):
        if img is None:
            return (None, False)
        if img.ndim != 2:
            grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            grayim = img
        return grayim, True

    def extract(self, img):
        # input img and output feature points&descriptors
        # if img is None:
        #     print("Load image error, Please check image_info topic")
        #     return
        if img is None:
            print("Load image error, Please check image_info topic")
            return
        # Get points and descriptors.
        kpts, desc = self.surf.detectAndCompute(img, None)
        pts = cv2.KeyPoint_convert(kpts)
        return pts.T, desc.T # [2,num_points], [121, num_points]
        # return pts.T, desc
  
class ORBPointMatchModel(BaseMatchModel):
    def _init(self, params=None):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, data):
        desc1 = np.array(data["descriptors0"], np.uint8)
        desc2 = np.array(data["descriptors1"], np.uint8)
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

class KnnPointMatchModel(BaseMatchModel):
    def _init(self, params=None):
        self.thresh = params["thresh"]
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, data):
        desc1 = np.array(data["descriptors0"], np.uint8)
        desc2 = np.array(data["descriptors1"], np.uint8)
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        
        pairs_of_matches = self.bf.knnMatch(desc1.T, desc2.T, k=2)
        matches = [x[0] for x in pairs_of_matches
                if len(x) > 1 and x[0].distance < self.thresh * x[1].distance]

        good_match = []
        for x in matches:
            good_match.append([x.queryIdx, x.trainIdx, x.distance])
        return np.array(good_match).T
    
class R2D2PointMatchModel(BaseMatchModel):
    def _init(self, params=None):
        self.thresh = params["thresh"]
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, data):
        desc1 = np.array(data["descriptors0"])
        desc2 = np.array(data["descriptors1"])
        dmat = np.dot(desc1.T, desc2)
        dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
        # Get NN indices and scores.
        idx = np.argmin(dmat, axis=1)
        scores = dmat[np.arange(dmat.shape[0]), idx]
        # Threshold the NN matches.
        keep = scores < self.thresh
        # Check if nearest neighbor goes both directions and keep those.
        idx2 = np.argmin(dmat, axis=0)
        keep_bi = np.arange(len(idx)) == idx2[idx]
        keep = np.logical_and(keep, keep_bi)
        idx = idx[keep]
        scores = scores[keep]
        # Get the surviving point indices.
        m_idx1 = np.arange(desc1.shape[1])[keep]
        m_idx2 = idx
        # Populate the final 3xN match data structure.
        matches = np.zeros((3, int(keep.sum())))
        matches[0, :] = m_idx1
        matches[1, :] = m_idx2
        matches[2, :] = scores
        return matches
    
class HarrisPointMatchModel(BaseMatchModel):
    def _init(self, params=None):
        pass

    def match(self, data):
        desc1 = np.array(data["descriptors0"], np.uint8)
        desc2 = np.array(data["descriptors1"], np.uint8)
        assert desc1.shape[0] == desc2.shape[0]
        if desc1.shape[1] == 0 or desc2.shape[1] == 0:
            return np.zeros((3, 0))
        # desc1 = desc1.tolist()
        # desc2 = desc2.tolist()
        # print(desc2)
        matches = harris.match_twosided(desc1.T, desc2.T)
        good_match = []
        for i in range(len(matches)):
            if matches[i] != -1:
                good_match.append([i, matches[i]])
        return np.array(good_match).T

