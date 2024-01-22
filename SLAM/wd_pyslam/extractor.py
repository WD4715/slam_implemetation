import cv2
import numpy as np

from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
from helpers import add_ones, poseRt, fundamentalToRt, normalize,  myjet #EssentialMatrixTransform,

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
        ret /= ret[2]
        return int(round(ret[0])), int(round(ret[1]))
        # return int(round(pt[0] + self.w // 2)), int(round(pt[1] + self.h // 2))
    def extractRt(self, E):        
        W = np.mat([[0, -1, 0], 
        [1, 0, 0], 
        [0, 0, 1]], dtype = float)

        # Lowe's ratio test 
        u, w, vt = np.linalg.svd(E)
        if np.linalg.det(u) < 0:
            u *= -1.
        if np.linalg.det(vt) < 0:
            vt *= -1.  
        R = np.dot(np.dot(u, W), vt)
        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(u, W.T), vt) 
        t = u[:, 2].reshape(3, 1)
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
                                residual_threshold=0.5, 
                                max_trials=100)
            ret = ret[inlier]
            
            # rotation matrix check logic
            pose = self.extractRt(model.params)

            # Focal length estimation
            # s, v, d = np.linalg.svd(model.params)
            # f_est = np.sqrt(2) / ((v[0] + v[1]) / 2)
            # self.f_est_avg.append(f_est)
            # print(f"f estimate : {f_est}, f_median{np.median(self.f_est_avg)}")
            # print(f"f average value : {np.mean(self.f_est_avg)}")
            # print(f"SVD result : {v}")


        self.last = {"kps" : kps, "des" : des}
        return ret, pose


def extractFeatures(img):
  orb = cv2.ORB_create()
  # detection
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

  # extraction
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
  kps, des = orb.compute(img, kps)

  # return pts and des
  return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def match_frames(f1, f2):

  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.des, f2.des, k=2)

  # Lowe's ratio test
  ret = []
  idx1, idx2 = [], []
  idx1s, idx2s = set(), set()

  for m,n in matches:
    if m.distance < 0.75*n.distance:
      p1 = f1.kps[m.queryIdx]
      p2 = f2.kps[m.trainIdx]

      # be within orb distance 32
      if m.distance < 20:
        # keep around indices
        # TODO: refactor this to not be O(N^2)
        if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
          idx1.append(m.queryIdx)
          idx2.append(m.trainIdx)
          idx1s.add(m.queryIdx)
          idx2s.add(m.trainIdx)
          ret.append((p1, p2))

  # no duplicates
  assert(len(set(idx1)) == len(idx1))
  assert(len(set(idx2)) == len(idx2))

  assert len(ret) >= 8
  ret = np.array(ret)
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)

  # fit matrix
  model, inliers = ransac((ret[:, 0], ret[:, 1]),
                          EssentialMatrixTransform,
                            min_samples=8, 
                            residual_threshold=5, 
                            max_trials=1000)
  print("Matches:  %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))
  return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)


class Frame(object):
  def __init__(self, img, K, pose=np.eye(4), tid=None, verts=None):
    self.K = np.array(K)
    self.pose = np.array(pose)

    if img is not None:
      self.h, self.w = img.shape[0:2]
      if verts is None:
        self.kps, self.des = extractFeatures(img)
      else:
        assert len(verts) < 256
        self.kps, self.des = verts, np.array(list(range(len(verts)))*32, np.uint8).reshape(32, len(verts)).T
      self.pts = [None]*len(self.kps)
    else:
      # fill in later
      self.h, self.w = 0, 0
      self.kps, self.des, self.pts = None, None, None

    # self.id = tid if tid is not None else mapp.add_frame(self)