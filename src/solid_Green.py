# green_fill.py

import cv2
import numpy as np

def solidGreen(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    kernel = np.ones((15,15), np.uint8)

    filled_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    result = np.full(
        img.shape,
        255,
        dtype=np.uint8
    )

    result[filled_mask > 0] = [0,255,0]

    return result, filled_mask