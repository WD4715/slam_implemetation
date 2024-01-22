import cv2

def process_frame(img):
    img = cv2.resize(img, (640, 480))
    cv2.imshow("image", img)