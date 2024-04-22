import numpy as np
from utils_line.sold2.model import SOLD2LineExtractModel
from utils_line.lcnn.model import LCNNLineExtractModel
from utils_line.sold2.model import WunschLinefeatureMatchModel
from utils_line.tplsd.model import TPLSDLineExtractModel
from utils_line.opencv_utils.model import KnnLineMatchModel

def create_lineextract_instance(params):
    extract_method = params["extract_method"]
    if extract_method == "sold2":
        params_dict = params["sold2"]
        params_dict["num_samples"] = params["num_samples"]
        return SOLD2LineExtractModel(params_dict)
    elif extract_method == "lcnn":
        params_dict = params["lcnn"]
        return LCNNLineExtractModel(params_dict)
    elif extract_method == "tplsd":
        params_dict = params["tplsd"]
        return TPLSDLineExtractModel(params_dict)
    else:
        raise ValueError("Line extract method {} is not supported!".format(extract_method))

def create_linematch_instance(params):
    match_method = params["match_method"]
    if match_method == "wunsch":
        params_dict = params["wunsch"]
        params_dict["num_samples"] = params["num_samples"]
        return WunschLinefeatureMatchModel(params_dict)
    elif match_method == "knn":
        params_dict = params["knn"]
        return KnnLineMatchModel(params_dict)
    else:
        raise ValueError("Line match method {} is not supported!".format(match_method))
