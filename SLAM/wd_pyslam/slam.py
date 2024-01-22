import sys
sys.path.append("/home/wondong/code/SLAM/wd_pyslam/build/g2opy/lib")
sys.path.append("/home/wondong/code/SLAM/wd_pyslam/build/pangolin")

import g2o
import pangolin

import cv2
import numpy as np

from extractor import FeatureExtractor, Frame, match_frames
from time import time

# from display import process_frame

W = 640
H = 480

# Camera intrinsic
F =190 # median value
# F =225 # mean value
K = np.array([[F, 0, W//2], 
              [0, F, H//2], 
              [0, 0, 1]])

frames = []
points = []

fe = FeatureExtractor(K)



IRt = np.eye(4)

# global map
class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []

    def display(self):
        for f in self.frames:

            print(f.pose)
            # print(f.id)
        for p in self.points:
            print(p.xyz)
   
# def display_map():

#     for f in frames:
#         print(f.pose)
#         print(f.id)
#     # for p in points:
#     #     print(p.xyz)


class Point(object):
    # Observations
    def __init__(self, mapp, loc):
        self.frames = []
        self.location = loc
        self.idxs = []

        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)
        

def triangulation(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3, :], pts1.T, pts2.T).T

mapp = Map() 
def process_frame(img):
    img = cv2.resize(img, (W, H))
    # mapp = Map() 

    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    
    kps1_idx, kps2_idx, Rt = match_frames(f1, f2)      
    kps1 = f1.kps[kps1_idx]
    kps2 = f2.kps[kps2_idx]   
    f1.pose = np.dot(Rt, f2.pose)   
    
    ## Triangulation
    pts4d = triangulation(IRt, Rt, kps1, kps2)
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.0) & (pts4d[:, 2] > 0.0)
    pts4d = pts4d[good_pts4d]
    
    # homogenous coordinate
    pts4d /= pts4d[:, 3:]    
    
    f1.pose = np.dot(Rt, f2.pose)

    for i, p in enumerate(pts4d):
        
        if not good_pts4d[i]:
            return
        pt = Point(mapp, p)
        pt.add_observation(f1, kps1_idx[i])
        pt.add_observation(f2, kps2_idx[i])
    
    for p1, p2 in zip(kps1, kps2):
        u1, v1 = int(p1[0]), int(p1[1])
        u2, v2 = int(p2[0]), int(p2[1])

        cv2.circle(img, (u1, v1), color = (0, 0, 255), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color = (255, 0, 0))
    cv2.imshow("image", img)
    mapp.frames.append(frame)
    
    # display_map()
    # Map().display()


if __name__ == "__main__":
    data_path = "/home/wondong/code/SLAM/wd_pyslam/data/front_undistorted_video.mp4"
    cap = cv2.VideoCapture(data_path)
    
    while cap.isOpened():
        start_time =time()
        ret, frame =  cap.read()
        
        if ret == True:
            process_frame(frame)
            end_time = time()

            print(f"fps : {1 / (end_time - start_time)}")


            if cv2.waitKey(1) & 0xff == ord("q"):
                break
        else:
            break