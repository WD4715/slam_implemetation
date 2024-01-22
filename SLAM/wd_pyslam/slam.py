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

fe = FeatureExtractor(K)



IRt = np.eye(4)

frames = []

class Point(object):
    def __init__(self, loc):
        self.frames = []
        self.location = loc
        self.idx = []
    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idx.append(idx)
        

def triangulation(pose1, pose2, pts1, pts2):
    # reject the without enough parallax and behind the camera
    pts4d = cv2.triangulatePoints(pose1[:3], pose2[:3, :], pts1.T, pts2.T).T
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0.0)
    pts4d = pts4d[good_pts4d]

    # homogenous coordinate
    pts4d /= pts4d[:, 3:]    
    return pts4d


def process_frame(img):
    img = cv2.resize(img, (W, H))
    
    frame = Frame(img, K)
    frames.append(frame)    
    if len(frames) <= 1:
        return
    kps1_idx, kps2_idx, Rt = match_frames(frames[-1], frames[-2])    
    kps1 = frames[-1].kps[kps1_idx]
    kps2 = frames[-2].kps[kps2_idx]
    # Rt = Rt[:3, :]
    # print(sum(Rt.diagonal()))
    # print(frames[-1].pose.shape)
    frames[-1].pose = np.dot(Rt, frames[-2].pose)
    
    # # reject the without enough parallax
    # pts4d = cv2.triangulatePoints(IRt[:3], Rt[:3, :], kps1.T, kps2.T).T
    # good_pts4d = np.abs(pts4d[:, 3]) > 0.005
    # pts4d = pts4d[good_pts4d]
    
    # # homogenous coordinate
    # pts4d /= pts4d[:, 3:]
    # # reject the behind the camera
    
    # good_pts4d = pts4d[:, 2] > 0.0
    # pts4d = pts4d[good_pts4d]
    pts4d = triangulation(IRt, Rt, kps1, kps2)

    for i, p in enumerate(pts4d):
        pt = Point(p)
        pt.add_observation(frames[-1], kps1_idx[i])
        pt.add_observation(frames[-2], kps2_idx[i])

    for p1, p2 in zip(kps1, kps2):
        u1, v1 = int(p1[0]), int(p1[1])
        u2, v2 = int(p2[0]), int(p2[1])

        cv2.circle(img, (u1, v1), color = (0, 0, 255), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color = (255, 0, 0))
    cv2.imshow("image", img)

    # pts, Rt = fe.extract(img)

    # if Rt is None:
    #     return
    # for p1, p2 in pts:
    #     u1, v1 = fe.denormalize(p1)
    #     u2, v2 = fe.denormalize(p2)

    #     cv2.circle(img, (u1, v1), color = (0, 0, 255), radius=3)
    #     cv2.line(img, (u1, v1), (u2, v2), color = (255, 0, 0))
    # cv2.imshow("image", img)


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