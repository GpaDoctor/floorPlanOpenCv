# import cv2
# import numpy as np

# def keepGreen(img):

#     hsv = cv2.cvtColor(
#         img,
#         cv2.COLOR_BGR2HSV
#     )

#     lower_green = np.array([45, 180, 180])
#     upper_green = np.array([85, 255, 255])

#     green_mask = cv2.inRange(
#         hsv,
#         lower_green,
#         upper_green
#     )

#     kernel = np.ones((5,5), np.uint8)

#     green_mask = cv2.morphologyEx(
#         green_mask,
#         cv2.MORPH_CLOSE,
#         kernel,
#         iterations=2
#     )

#     result = np.full_like(img, 255)

#     result[green_mask > 0] = img[green_mask > 0]

#     return result

# import cv2
# import numpy as np

# def keepGreen(img):

#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # Ignore black, white, gray
#     color_mask = cv2.inRange(
#         hsv,
#         np.array([0, 50, 50]),
#         np.array([179, 255, 255])
#     )

#     hue_values = hsv[:, :, 0][color_mask > 0]

#     if len(hue_values) == 0:
#         return np.full_like(img, 255)

#     # Find dominant hue
#     hist = np.bincount(hue_values)
#     dominant_hue = np.argmax(hist)

#     print("Dominant hue:", dominant_hue)

#     # Create mask around dominant hue
#     lower = np.array([
#         max(0, dominant_hue - 10),
#         50,
#         50
#     ])

#     upper = np.array([
#         min(179, dominant_hue + 10),
#         255,
#         255
#     ])

#     dominant_mask = cv2.inRange(hsv, lower, upper)

#     # Clean up
#     kernel = np.ones((5,5), np.uint8)
#     dominant_mask = cv2.morphologyEx(
#         dominant_mask,
#         cv2.MORPH_CLOSE,
#         kernel,
#         iterations=2
#     )

#     # Output image
#     result = np.full_like(img, 255)
#     result[dominant_mask > 0] = img[dominant_mask > 0]

#     return result
import cv2
import numpy as np

def keepGreen(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Ignore white, gray, black
    valid_mask = (
        (hsv[:, :, 1] > 50) &
        (hsv[:, :, 2] > 50)
    )

    hues = hsv[:, :, 0][valid_mask]

    if len(hues) == 0:
        return np.full_like(img, 255)

    # Find color with largest coverage
    hist = np.bincount(hues.astype(np.uint8))

    dominant_hue = np.argmax(hist)

    print(f"Dominant Hue = {dominant_hue}")

    # Create mask for only that color
    hue_tolerance = 1

    lower_hue = max(0, dominant_hue - hue_tolerance)
    upper_hue = min(179, dominant_hue + hue_tolerance)

    dominant_mask = cv2.inRange(
        hsv,
        np.array([lower_hue, 50, 50]),
        np.array([upper_hue, 255, 255])
    )

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)

    dominant_mask = cv2.morphologyEx(
        dominant_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Find connected regions of that color
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dominant_mask,
        connectivity=8
    )

    if num_labels <= 1:
        return np.full_like(img, 255)

    # Keep ONLY the largest connected region
    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    largest_mask = np.zeros_like(dominant_mask)

    largest_mask[labels == largest_label] = 255

    print(
        "Largest Area =",
        stats[largest_label, cv2.CC_STAT_AREA]
    )

    # Create output image
    result = np.full_like(img, 255)

    result[largest_mask > 0] = img[largest_mask > 0]

    return result
    # return result, dominant_hue