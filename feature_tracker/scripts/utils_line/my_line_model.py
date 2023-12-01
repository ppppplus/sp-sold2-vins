import numpy as np
from utils_line.sold2.model import SOLD2LineExtractModel
from utils_line.sold2.model import WunschLinefeatureMatchModel

def create_lineextract_instance(params):
    extract_method = params["extract_method"]
    if extract_method == "sold2":
        params_dict = params["sold2"]
        params_dict["num_samples"] = params["num_samples"]
        return SOLD2LineExtractModel(params_dict)
    else:
        raise ValueError("Line extract method {} is not supported!".format(extract_method))

def create_linematch_instance(params):
    match_method = params["match_method"]
    if match_method == "wunsch":
        params_dict = params["wunsch"]
        params_dict["num_samples"] = params["num_samples"]
        return WunschLinefeatureMatchModel(params_dict)
    else:
        raise ValueError("Line match method {} is not supported!".format(match_method))
