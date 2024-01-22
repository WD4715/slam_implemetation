import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

class FeatureExtractor(object):
    
    def __init__(self):
        self.orb = cv2.ORB_create(100)
        self.last = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
    def extract(self, img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        
        # extraction
        kps = [cv2.KeyPoint(x = f[0][0], y = f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        
        # matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last["des"], k= 2)
            # matches = zip([kps[m.queryIdx] for m in matches], [self.last['kps'][m.trainIdx] for m in matches])
            for m, n in matches:
                if m.distance < 0.75 *n.distance:
                    kps1, kps2 = kps[m.queryIdx].pt, self.last['kps'][m.trainIdx].pt
                    ret.append((kps1, kps2))
        # filtering
        if len(ret)> 0 :
            ret = np.array(ret)
            
            print(ret.shape)
            model, inlier = ransac((ret[:, 0], ret[:, 1]), 
                                FundamentalMatrixTransform, 
                                min_samples=8, 
                                residual_threshold=1, 
                                max_trials=100)
            ret = ret[inlier]
            print(f"inliers : {sum(inlier)}")
        self.last = {"kps" : kps, "des" : des}
        return ret
