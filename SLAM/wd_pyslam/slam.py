import cv2

from display import process_frame

if __name__ == "__main__":
    data_path = "/home/wondong/code/SLAM/wd_pyslam/data/front_undistorted_video.mp4"
    cap = cv2.VideoCapture(data_path)
    while cap.isOpened():

        ret, frame =  cap.read()

        if ret == True:
            process_frame(frame)


            if cv2.waitKey(1) & 0xff == ord("q"):
                break
        else:
            break