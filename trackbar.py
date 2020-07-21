import numpy as np
import cv2


def nothing(x):
    pass


def initialize_trackbar(intial_trackbar_vals):
    cv2.namedWindow("Trackbar")
    cv2.resizeWindow("Trackbar", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbar", intial_trackbar_vals[0], 1500, nothing)
    cv2.createTrackbar("Height Top", "Trackbar", intial_trackbar_vals[1], 1500, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbar", intial_trackbar_vals[2], 1500, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbar", intial_trackbar_vals[3], 1500, nothing)


def val_trackbar(image):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbar")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbar")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbar")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbar")

    src = np.float32([(image.shape[1] - widthBottom, heightBottom), (image.shape[1] - widthTop, heightTop), (widthTop, heightTop), (widthBottom, heightBottom)])

    return src


def draw_points(img, src):
    for x in range(4):
        cv2.circle(img, (int(src[x][0]), int(src[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img
