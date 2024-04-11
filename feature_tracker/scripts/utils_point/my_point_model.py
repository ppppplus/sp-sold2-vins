import numpy as np
from utils_point.superpoint.model import SuperpointPointExtractModel, NnmPointMatchModel
from utils_point.r2d2.model import R2D2PointExtractModel
from utils_point.opencv_utils.model import ORBPointExtractModel, KnnPointMatchModel, FASTPointExtractModel, HarrisPointExtractModel, HarrisPointMatchModel, SIFTPointExtractModel, SURFPointExtractModel, R2D2PointMatchModel
# from utils_point.superpoint.trt_model import TrtSuperpointPointExtractModel
from utils_point.superglue.model import SuperGlueMatchModel

def create_pointextract_instance(params):
    extract_method = params["extract_method"]
    if extract_method == "superpoint":
        return SuperpointPointExtractModel(params["superpoint"])
    # if extract_method == "superpoint_trt":
    #     return TrtSuperpointPointExtractModel(params["superpoint_trt"])
    elif extract_method == "orb":
        return ORBPointExtractModel(params)
    elif extract_method == "sift":
        return SIFTPointExtractModel(params)
    elif extract_method == "surf":
        return SURFPointExtractModel(params)
    elif extract_method == "r2d2":
        return R2D2PointExtractModel(params)
    else:
        raise ValueError("Extract method {} is not supported!".format(extract_method))

def create_pointmatch_instance(params):
    match_method = params["match_method"]
    if match_method == "nnm":
        return NnmPointMatchModel(params["nnm"])
    elif match_method == "superglue":
        return SuperGlueMatchModel(params["superglue"])
    elif match_method == "knn":
        return KnnPointMatchModel(params["knn"])
    elif match_method == "r2d2":
        return R2D2PointMatchModel(params["r2d2"])
    elif match_method == "harris":
        return HarrisPointMatchModel(params)
    else:
        raise ValueError("Match method {} is not supported!".format(match_method))
