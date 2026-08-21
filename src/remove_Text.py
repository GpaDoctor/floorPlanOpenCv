import cv2
import numpy as np

def removeText(img):

    # Convert to grayscale only for analysis
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Binary image for connected components
    _, bw = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Connected components
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            bw,
            connectivity=8
        )

    keep_mask = np.zeros_like(bw)

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        # Keep larger objects
        if area > 5000:
            keep_mask[
                labels == i
            ] = 255

    # Cleanup
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    keep_mask = cv2.morphologyEx(
        keep_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # White canvas, same format as input
    result = np.full_like(
        img,
        255
    )

    # Copy original pixels back
    result[keep_mask > 0] = img[keep_mask > 0]

    return result