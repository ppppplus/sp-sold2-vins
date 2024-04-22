import cv2
import numpy as np
from utils.base_model import BaseExtractModel, BaseMatchModel

class KnnLineMatchModel(BaseMatchModel):
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
        # matches = [x[0] for x in pairs_of_matches
        #         if len(x) > 1 and x[0].distance < self.thresh * x[1].distance]
        matches = [m for m, n in pairs_of_matches if m.distance < 20 and m.distance < n.distance * 0.7]
        matches = sorted(matches, key=lambda x: x.distance)

        matches_index = []
        for x in matches:
            matches_index.append([x.queryIdx, x.trainIdx])
        return np.array(matches_index)