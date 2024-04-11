import numpy as np
from utils.base_model import BaseMatchModel
from . import superglue

class SuperGlueMatchModel(BaseMatchModel):
    def _init(self, params):
        self.model = superglue.SuperGlue(params)
    def match(self, data):
        matches = self.model(data)
        return matches