import sys
sys.path.append("/home/wondong/code/SLAM/wd_pyslam/build/g2opy/lib")
sys.path.append("/home/wondong/code/SLAM/wd_pyslam/build/pangolin")

import g2o
import cv2
import numpy as np
import pangolin
import OpenGL.GL as gl

from extractor import FeatureExtractor, Frame, match_frames
from time import time
from multiprocessing import Process, Queue

# from display import process_frame

W = 640
H = 480

# Camera intrinsic
F =190 # median value
# F =225 # mean value
K = np.array([[F, 0, W//2], 
              [0, F, H//2], 
              [0, 0, 1]])
K_inv = np.linalg.inv(K)

frames = []
points = []

fe = FeatureExtractor(K)



IRt = np.eye(4)

# global map
class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.q = Queue()
        self.state = None
        
        p = Process(target = self.viewer_thread, args = (self.q, ))
        p.daemon = True
        p.start()
        
        
        '''### Create viewer | Multi-Processing
        self.q = Queue()
        self.viewer = Process(target=self.viewer_thread, args=(self.q, ))
        self.viewer.daemon = True
        self.viewer.start()
        '''
    def viewer_thread(self, q):
        self.viewer_init()
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self):
        
        # params = pangolin.Params()
        # 직접 전달하는 방법
        params = pangolin.Params()
        # params["position"] = (-500, -500)


        pangolin.CreateWindowAndBind("Main", 256, 256, params)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(0, 0, 20, 
                               0, 0, 0, 
                               0, -1, 0)) 
            # pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        self.handler = pangolin.Handler3D(self.scam)
        
        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self, q):
        if self.state is None or q.empty():
            self.state = q.get()
        # turn state to points
        # ppts = np.array([d[:3, 3]for d in self.state[0]])
        spts = np.array(self.state[1])
        
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawCameras(self.state[0])
        # for pose in self.state[0]:
        #     pangolin.glDrawFrustrum(K_inv, 2, 2, pose, 1)

        # gl.glPointSize(10)
        # 
        
        # pangolin.DrawPoints(ppts)    
        
        gl.glPointSize(4)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(spts)    

        pangolin.FinishFrame()

    def display_map(self):
        poses = []
        pts = []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        # self.state = poses, pts
        self.q.put((poses, pts))
        # self.viewer_refresh(self.q)


            

class Point(object):
    # Observations
    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []

        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)
        

def triangulation(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret

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
    pts4d = triangulation(f1.pose, f2.pose, kps1, kps2)
    pts4d /= pts4d[:, 3:]    
    
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0.0)
    pts4d = pts4d[good_pts4d]
    if len(pts4d) == 0:
        return

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
    mapp.display_map()


if __name__ == "__main__":
    data_path = "/home/wondong/code/SLAM/wd_pyslam/data/video.mp4"
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