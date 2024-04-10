#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%


import argparse
import glob
import numpy as np
import os
import time

import cv2
import torch
import yaml

from pylsd import lsd

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

class MyLineExtractModel:
  def __init__(self):
    
    # 创建LSD检测器对象
    # self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    self.lsd = cv2.line_descriptor_LSDDetector.createLSDDetector()


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
    # grayim = (grayim.astype('float32') / 255.)
    return grayim, True

  def extract_line(self, img):
    # input img and output feature lines
    # This class runs the LSD and processes its outputs.
    grayim, status = self.process_image(img)
    if status is False:
      print("Load image error, Please check image_info topic")
      return
    # keylines = cv2.line_descriptor.KeyLine
    # keylines = lsd(grayim)
    # keylines, _ = self.lsd.detect(grayim)
    # keylines, descriptors = self.lsd.compute(grayim, keylines)
    keylines = lsd(grayim, scale=0.5)
    print('keylines:', keylines)
    # keylines, descriptors = self.lsd.compute(grayim, keylines)
    
    return keylines
  
  def draw_line(self, img, keylines):
    # 绘制检测到的线段
    # line_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # self.lsd.drawSegments(output_image, lines)
    # line_img = cv2.drawKeylines(line_img, keylines, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    line_img = img.copy()
    for i in range(keylines.shape[0]):
        pt1 = (int(keylines[i, 0]), int(keylines[i, 1]))
        pt2 = (int(keylines[i, 2]), int(keylines[i, 3]))
        width = keylines[i, 4]
        cv2.line(line_img, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))

    return line_img

class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, height, width, skip, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv2.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv2.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        self.maxlen = len(self.listing)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

  def read_image(self, img, img_size):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    # grayim = cv2.imread(impath, 0)
    grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if grayim is None:
      raise Exception('Error reading image')
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, False)
    if self.camera:
      ret, input_image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None, False)
      if self.video_file:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv2.INTER_AREA)
      input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
      input_image = input_image.astype('float')/255.0
    else:
      image_file = self.listing[self.i]
      input_image = self.read_image(image_file, self.sizer)
    # Increment internal counter.
    self.i = self.i + 1
    input_image = input_image.astype('float32')
    return (input_image, True)


# if __name__ == '__main__':

#   yamlPath = 'config.yaml'
#   pointmodel = MyPointModel(yamlPath)
#   cap = cv2.VideoCapture('assets/nyu_snippet.mp4')
#   ret, frame = cap.read()
#   cv2.imshow('frame', frame)
#   cv2.waitKey(0)

#   if ret:
#     pts = pointmodel.extract_point(frame)
  # with open(yamlPath,'rb') as f:
  #     # yaml文件通过---分节，多个节组合成一个列表
  #     params = yaml.load(f, Loader=yaml.FullLoader)
  # vs = VideoStreamer(params["input"], params["camid"], params["H"], params["W"], params["skip"], params["img_glob"])
  # # This class runs the SuperPoint network and processes its outputs.
  # fe = SuperPointFrontend(weights_path=params["weights_path"],
  #                         nms_dist=params["nms_dist"],
  #                         conf_thresh=params["conf_thresh"],
  #                         nn_thresh=params["nn_thresh"],
  #                         cuda=params["cuda"])
  # # This class helps merge consecutive point matches into tracks.
  # tracker = PointTracker(params["max_length"], nn_thresh=fe.nn_thresh)
  # # Create a window to display the demo.
  # if not params["no_display"]:
  #   win = 'SuperPoint Tracker'
  #   cv2.namedWindow(win)
  # else:
  #   print('Skipping visualization, will not show a GUI.')

  # # Font parameters for visualizaton.
  # font = cv2.FONT_HERSHEY_DUPLEX
  # font_clr = (255, 255, 255)
  # font_pt = (4, 12)
  # font_sc = 0.4

  # # Create output directory if desired.
  # # if params["write"]:
  # #   print('==> Will write outputs to %s' % params["write_dir"])
  # #   if not os.path.exists(params["write_dir"]):
  # #     os.makedirs(params["write_dir"])

  # while True:

  #   start = time.time()

  #   # Get a new image.
  #   img, status = vs.next_frame()
  #   cv2.imshow(img)
  #   cv2.waitKey(0)
  #   if status is False:
  #     break

  #   # Get points and descriptors.
  #   start1 = time.time()
  #   pts, desc, heatmap = fe.run(img)
  #   end1 = time.time()

  #   # Add points and descriptors to the tracker.
  #   tracker.update(pts, desc)

  #   # Get tracks for points which were match successfully across all frames.
  #   tracks = tracker.get_tracks(params["min_length"])

  #   # Primary output - Show point tracks overlayed on top of input image.
  #   out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
  #   tracks[:, 1] /= float(fe.nn_thresh) # Normalize track scores to [0,1].
  #   tracker.draw_tracks(out1, tracks)
  #   if params["show_extra"]:
  #     cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)

  #   # Extra output -- Show current point detections.
  #   out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
  #   for pt in pts.T:
  #     pt1 = (int(round(pt[0])), int(round(pt[1])))
  #     cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
  #   cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

  #   # Extra output -- Show the point confidence heatmap.
  #   if heatmap is not None:
  #     min_conf = 0.001
  #     heatmap[heatmap < min_conf] = min_conf
  #     heatmap = -np.log(heatmap)
  #     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
  #     out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
  #     out3 = (out3*255).astype('uint8')
  #   else:
  #     out3 = np.zeros_like(out2)
  #   cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

  #   # Resize final output.
  #   if params["show_extra"]:
  #     out = np.hstack((out1, out2, out3))
  #     out = cv2.resize(out, (3*params["display_scale"]*params["W"], params["display_scale"]*params["H"]))
  #   else:
  #     out = cv2.resize(out1, (params["display_scale"]*params["W"], params["display_scale"]*params["H"]))

  #   # Display visualization image to screen.
  #   if not params["no_display"]:
  #     cv2.imshow(win, out)
  #     key = cv2.waitKey(params["waitkey"]) & 0xFF
  #     if key == ord('q'):
  #       print('Quitting, \'q\' pressed.')
  #       break

  #   # Optionally write images to disk.
  #   # if params["write"]:
  #   #   out_file = os.path.join(params["write_dir"], 'frame_%05d.png' % vs.i)
  #   #   print('Writing image to %s' % out_file)
  #   #   cv2.imwrite(out_file, out)

  #   end = time.time()
  #   net_t = (1./ float(end1 - start))
  #   total_t = (1./ float(end - start))
  #   if params["show_extra"]:
  #     print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS).'\
  #           % (vs.i, net_t, total_t))

  # # Close any remaining windows.
  # cv2.destroyAllWindows()
