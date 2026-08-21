import cv2
import numpy as np

def keepGreen(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    lower_green = np.array([45, 180, 180])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    kernel = np.ones((5,5), np.uint8)

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    result = np.full_like(img, 255)

    result[green_mask > 0] = img[green_mask > 0]

    return result
