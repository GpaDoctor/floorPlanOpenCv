# walls.py

import cv2

def extractWalls(img):

    # Convert to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove small noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    clean = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    # Thicken wall structures
    wall_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5)
    )

    walls = cv2.dilate(
        clean,
        wall_kernel,
        iterations=1
    )

    # Black walls on white background
    inverted = cv2.bitwise_not(walls)

    return gray, thresh, inverted