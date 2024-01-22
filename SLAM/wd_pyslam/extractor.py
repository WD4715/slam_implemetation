import cv2
import numpy as np

class FeatureExtractor(object):
    
    def __init__(self):
        self.orb = cv2.ORB_create(10000)
        self.last = None
        self.bf = cv2.BFMatcher()
        
    def extract(self, img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        
        # extraction
        kps = [cv2.KeyPoint(x = f[0][0], y = f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        
        # matching
        matches = None
        self.last = {"kps" : kps, "des" : des}
        if self.last is not None:
            matches = self.bf.match(des, self.last["des"])
        return kps, des, matches
