import torch
import cv2
import numpy as np
from utils.base_model import BaseExtractModel, BaseMatchModel
from .utils.reconstruct import TPS_line
from .lbdmod.build import pylbd

from .utils.utils import load_model
from .modeling.TP_Net import Res160, Res320
from .modeling.Hourglass import HourglassNet

import os

class TplsdDetect:
  def __init__(self, modeluse):
    current_path = os.path.dirname(__file__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
      raise EOFError('cpu version for training is not implemented.')
    print('Using device: ', device)
    self.head = {'center': 1, 'dis': 4, 'line': 1}
    if modeluse == 'tp320':
        self.model = load_model(Res320(self.head), os.path.join(current_path, 'pretraineds/Res320.pth'))
        self.in_res = (320, 320)
    elif modeluse == 'tplite':
        self.model = load_model(Res160(self.head), os.path.join(current_path, 'pretraineds/Res160.pth'))
        self.in_res = (320, 320)
    elif modeluse == 'tp512':
        self.model = load_model(Res320(self.head), os.path.join(current_path, 'pretraineds/Res512.pth'))
        self.in_res = (512, 512)
    elif modeluse == 'hg':
        self.model = load_model(HourglassNet(self.head), os.path.join(current_path, 'pretraineds/HG128.pth'))
        self.in_res = (512, 512)
    else:
        raise EOFError('Please appoint the correct model (option: tp320, tplite, tp512, hg). ')

    self.model = self.model.cuda().eval()


  def getlines(self, outputs, H, W, H_img, W_img):
    output = outputs[-1]
    lines, start_point, end_point, pos, endtime = TPS_line(output, 0.25, 0.5, H, W)
    W_ = W_img / W
    H_ = H_img / H
    lines[:, [0, 2]] *= W_
    lines[:, [1, 3]] *= H_
    return lines

  def detect_tplsd(self, img):
    H_img, W_img = img.shape[:2]
    inp = cv2.resize(img, self.in_res)
    H, W, C = inp.shape
    hsv = cv2.cvtColor(inp, cv2.COLOR_BGR2HSV)
    imgv0 = hsv[..., 2]
    imgv = cv2.resize(imgv0, (0, 0), fx=1. / 4, fy=1. / 4, interpolation=cv2.INTER_LINEAR)
    imgv = cv2.GaussianBlur(imgv, (5, 5), 3)
    imgv = cv2.resize(imgv, (W, H), interpolation=cv2.INTER_LINEAR)
    imgv = cv2.GaussianBlur(imgv, (5, 5), 3)

    imgv1 = imgv0.astype(np.float32) - imgv + 127.5
    imgv1 = np.clip(imgv1, 0, 255).astype(np.uint8)
    hsv[..., 2] = imgv1
    inp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    inp = (inp.astype(np.float32) / 255.)
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
      outputs = self.model(inp)
    lines = self.getlines(outputs, H, W, H_img, W_img)
    return lines

class TPLSDLineExtractModel(BaseExtractModel):
    def _init(self, params):
        # self.device = 'cuda' if torch.cuda.is_available() is True else 'cpu'
        self.tplsd = TplsdDetect(params["model"])

    def extract(self, img):
        if img.ndim == 2:
            im = np.repeat(img[:, :, None], 3, 2)
        kls = self.tplsd.detect_tplsd(im)

        desc = pylbd.describe_with_lbd(img, kls, 1, 1.44)
        vecline = kls.reshape((-1,2,2))[:,::-1,::-1]
        return vecline, desc.T  # num_lines*2*2; desc_dim*num_lines;
    
