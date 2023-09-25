import numpy as np
from utils_pl.model_bp import SPSOLD2ExtractModel
from utils_point.superpoint.model import NnmPointMatchModel
from utils_line.sold2.model import WunschLinefeatureMatchModel

def create_plextract_instance(params):
    extract_method = params["extract_method"]
    if extract_method == "sp-sold2":
        return SPSOLD2ExtractModel(params["sp-sold2"])
    else:
        raise ValueError("Extract method {} is not supported!".format(extract_method))

def create_pointmatch_instance(params):
    match_method = params["match_method"]
    if match_method == "nnm":
        return NnmPointMatchModel(params["nnm"])
    else:
        raise ValueError("Match method {} is not supported!".format(match_method))

def create_linematch_instance(params):
    match_method = params["match_method"]
    if match_method == "wunsch":
        params_dict = params["wunsch"]
        params_dict["num_samples"] = params["num_samples"]
        return WunschLinefeatureMatchModel(params_dict)
    else:
        raise ValueError("Line match method {} is not supported!".format(match_method))
