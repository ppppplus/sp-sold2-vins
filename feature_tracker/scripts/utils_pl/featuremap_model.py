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
import math
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
        
class SPSOLD2Model():
    def __init__(self, params):
        self.params = params
        self.heatmap_refine_cfg = self.params["heatmap_refine_cfg"]
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
        heatmap = nn.functional.softmax(heatmap, dim=1)[:, 1:, :, :]
        if self.heatmap_refine_cfg["mode"] == "global":
                heatmap = self.refine_heatmap(
                    heatmap, 
                    self.heatmap_refine_cfg["ratio"],
                    self.heatmap_refine_cfg["valid_thresh"]
                )
        elif self.heatmap_refine_cfg["mode"] == "local":
            heatmap = self.refine_heatmap_local(
                heatmap, 
                self.heatmap_refine_cfg["num_blocks"],
                self.heatmap_refine_cfg["overlap_ratio"],
                self.heatmap_refine_cfg["ratio"],
                self.heatmap_refine_cfg["valid_thresh"]
            )
        #########################################################
        featuremap = {
            "heatmap": heatmap.squeeze(0).cpu().numpy(),
            "junction": junctions.squeeze().cpu().numpy(),
            "coarse_desc": coarse_desc.squeeze().cpu().numpy()
        }
        
        return featuremap
        
    
    def refine_heatmap(self, heatmap, ratio=0.2, valid_thresh=1e-2):
        """ Global heatmap refinement method. """
        # Grab the top 10% values
        heatmap_values = heatmap[heatmap > valid_thresh]
        sorted_values = torch.sort(heatmap_values, descending=True)[0]
        top10_len = math.ceil(sorted_values.shape[0] * ratio)
        max20 = torch.mean(sorted_values[:top10_len])
        heatmap = torch.clamp(heatmap / max20, min=0., max=1.)
        return heatmap
    
    def refine_heatmap_local(self, heatmap, num_blocks=5, overlap_ratio=0.5,
                             ratio=0.2, valid_thresh=2e-3):
        """ Local heatmap refinement method. """
        # Get the shape of the heatmap
        H, W = heatmap.shape
        increase_ratio = 1 - overlap_ratio
        h_block = round(H / (1 + (num_blocks - 1) * increase_ratio))
        w_block = round(W / (1 + (num_blocks - 1) * increase_ratio))

        count_map = torch.zeros(heatmap.shape, dtype=torch.int,
                                device=heatmap.device)
        heatmap_output = torch.zeros(heatmap.shape, dtype=torch.float,
                                     device=heatmap.device)
        # Iterate through each block
        for h_idx in range(num_blocks):
            for w_idx in range(num_blocks):
                # Fetch the heatmap
                h_start = round(h_idx * h_block * increase_ratio)
                w_start = round(w_idx * w_block * increase_ratio)
                h_end = h_start + h_block if h_idx < num_blocks - 1 else H
                w_end = w_start + w_block if w_idx < num_blocks - 1 else W

                subheatmap = heatmap[h_start:h_end, w_start:w_end]
                if subheatmap.max() > valid_thresh:
                    subheatmap = self.refine_heatmap(
                        subheatmap, ratio, valid_thresh=valid_thresh)
                
                # Aggregate it to the final heatmap
                heatmap_output[h_start:h_end, w_start:w_end] += subheatmap
                count_map[h_start:h_end, w_start:w_end] += 1
        heatmap_output = torch.clamp(heatmap_output / count_map,
                                     max=1., min=0.)

        return heatmap_output
    
def weight_init(m):
    """ Weight initialization function. """
    # Conv2D
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    # Batchnorm
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    # Linear
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    else:
        pass