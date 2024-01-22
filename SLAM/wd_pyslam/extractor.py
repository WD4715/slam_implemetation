import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
  if len(x.shape) == 1:
    return np.concatenate([x,np.array([1.0])], axis=0)
  else:
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


class FeatureExtractor(object):
    
    def __init__(self, K):
        self.orb = cv2.ORB_create(100)
        self.last = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        self.f_est_avg = []

    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    def denormalize(self, pt):
        
        ret =  np.dot(self.K, np.array([pt[0], pt[1], 1]))
        
        return int(round(ret[0])), int(round(ret[1]))
        # return int(round(pt[0] + self.w // 2)), int(round(pt[1] + self.h // 2))
    def extractRt(self, E):
                W = np.mat([[0, -1, 0], 
                [1, 0, 0], 
                [0, 0, 1]], dtype = float)

                u, w, vt = np.linalg.svd(E)
                if np.linalg.det(u) < 0:
                    u *= -1.
                if np.linalg.det(vt) < 0:
                    vt *= -1.  
                R = np.dot(np.dot(u, W), vt)
                if np.sum(R.diagonal()) < 0:
                    R = np.dot(np.dot(u, W.T), vt) 
                t = u[:, 2].reshape(3, 1)

                print(R.shape)
                print(t.shape)
                pose = np.concatenate([R, t], axis = 1)
                
                return pose

        
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
        pose = None
        if len(ret)> 0 :

            # normalization for fundamental matrix
            ret = np.array(ret)
            ret[:, 0, :] = self.normalize(ret[:,0, :])
            ret[:, 1, :] = self.normalize(ret[:,1, :])


            model, inlier = ransac((ret[:, 0], ret[:, 1]), 
                                # FundamentalMatrixTransform, 
                                EssentialMatrixTransform,
                                min_samples=8, 
                                residual_threshold=0.06, 
                                max_trials=100)
            ret = ret[inlier]
            
            # rotation matrix check logic
            pose = self.extractRt(model.params)

            print("#"*50)
            print("Rotation matrix : ")
            print(np.sum(pose.diagonal()))
            print(pose)
            

            # Focal length estimation
            # s, v, d = np.linalg.svd(model.params)
            # f_est = np.sqrt(2) / ((v[0] + v[1]) / 2)
            # self.f_est_avg.append(f_est)
            # print(f"f estimate : {f_est}, f_median{np.median(self.f_est_avg)}")
            # print(f"f average value : {np.mean(self.f_est_avg)}")

            # print(f"SVD result : {v}")


        self.last = {"kps" : kps, "des" : des}
        return ret, pose
