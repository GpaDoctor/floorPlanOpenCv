# # remove_green.py

# import cv2
# import numpy as np

# def removeGreen(img):

#     hsv = cv2.cvtColor(
#         img,
#         cv2.COLOR_BGR2HSV
#     )

#     lower_green = np.array([35, 40, 40])
#     upper_green = np.array([90, 255, 255])

#     green_mask = cv2.inRange(
#         hsv,
#         lower_green,
#         upper_green
#     )

#     kernel = np.ones((5, 5), np.uint8)

#     green_mask = cv2.morphologyEx(
#         green_mask,
#         cv2.MORPH_CLOSE,
#         kernel,
#         iterations=2
#     )

#     gray = cv2.cvtColor(
#         img,
#         cv2.COLOR_BGR2GRAY
#     )

#     result_gray = np.clip(
#         gray * 1.25,
#         0,
#         255
#     ).astype(np.uint8)

#     result = cv2.cvtColor(
#         result_gray,
#         cv2.COLOR_GRAY2BGR
#     )

#     result[green_mask > 0] = (
#         255,
#         255,
#         255
#     )

#     dark_mask = gray < 130

#     for c in range(3):
#         result[:, :, c][dark_mask] = gray[dark_mask]

#     return result

import cv2
import numpy as np

def removeGreen(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Text tends to be dark
    text_mask = gray < 130

    result = np.full_like(
        img,
        255
    )

    result[text_mask] = (
        0,
        0,
        0
    )

    return result
