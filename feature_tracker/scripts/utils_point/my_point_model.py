import numpy as np
from utils_point.superpoint.model import SuperpointPointExtractModel, NnmPointMatchModel
from utils_point.opencv_utils.model import ORBPointExtractModel, KnnPointMatchModel, FASTPointExtractModel
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
    elif extract_method == "fast":
        return FASTPointExtractModel(params)
    else:
        raise ValueError("Extract method {} is not supported!".format(extract_method))

def create_pointmatch_instance(params):
    match_method = params["match_method"]
    if match_method == "nnm":
        return NnmPointMatchModel(params["nnm"])
    elif match_method == "superglue":
        return SuperGlueMatchModel(params["superglue"])
    elif match_method == "knn":
        return KnnPointMatchModel(params)
    else:
        raise ValueError("Match method {} is not supported!".format(match_method))
