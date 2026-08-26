# import cv2
# import numpy as np

# def removeText(img):

#     # Convert to grayscale only for analysis
#     gray = cv2.cvtColor(
#         img,
#         cv2.COLOR_BGR2GRAY
#     )

#     # Binary image for connected components
#     _, bw = cv2.threshold(
#         gray,
#         180,
#         255,
#         cv2.THRESH_BINARY_INV
#     )

#     # Connected components
#     num_labels, labels, stats, centroids = \
#         cv2.connectedComponentsWithStats(
#             bw,
#             connectivity=8
#         )

#     keep_mask = np.zeros_like(bw)

#     # for i in range(1, num_labels):

#     #     area = stats[
#     #         i,
#     #         cv2.CC_STAT_AREA
#     #     ]

#     #     # Keep larger objects
#     #     if area > 5000:
#     #         keep_mask[
#     #             labels == i
#     #         ] = 255

#     # for i in range(1, num_labels):

#     #     area = stats[i, cv2.CC_STAT_AREA]
#     #     w = stats[i, cv2.CC_STAT_WIDTH]
#     #     h = stats[i, cv2.CC_STAT_HEIGHT]

#     #     # Remove things that look like text
#     #     if area < 1500 and h < 50:
#     #         continue

#     #     keep_mask[labels == i] = 255

#     for i in range(1, num_labels):

#         area = stats[i, cv2.CC_STAT_AREA]
#         w = stats[i, cv2.CC_STAT_WIDTH]
#         h = stats[i, cv2.CC_STAT_HEIGHT]

#         aspect = max(w, h) / max(1, min(w, h))

#         # Likely text
#         if area < 1500 and h < 50 and aspect < 10:
#             continue

#         keep_mask[labels == i] = 255

#     # Cleanup
#     kernel = np.ones(
#         (2, 2),
#         np.uint8
#     )

#     keep_mask = cv2.morphologyEx(
#         keep_mask,
#         cv2.MORPH_CLOSE,
#         kernel,
#         iterations=1
#     )

#     # White canvas, same format as input
#     result = np.full_like(
#         img,
#         255
#     )

#     # Copy original pixels back
#     result[keep_mask > 0] = img[keep_mask > 0]

#     return result

import cv2
import numpy as np

def blank_page(img):
    return np.full_like(img, 255)


def removeText(
    img,
    max_component_ratio=0.5,
    max_changed_ratio=0.7,
    min_removed_ratio=0.95
):
    """
    Remove likely text while preventing catastrophic failures.

    Returns:
        result_img
    """

    h_img, w_img = img.shape[:2]
    total_pixels = h_img * w_img

    # Convert to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Binary image
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

    # --------------------------------------------------
    # SAFETY CHECK #1
    # Giant component means threshold probably failed
    # --------------------------------------------------
    if num_labels > 1:
        largest_area = np.max(
            stats[1:, cv2.CC_STAT_AREA]
        )

        if largest_area > total_pixels * max_component_ratio:
            print(
                f"[removeText] FAIL: giant component "
                f"({largest_area:,} pixels)"
            )
            return blank_page(img), 1

    keep_mask = np.zeros_like(bw)

    # --------------------------------------------------
    # Component filtering
    # --------------------------------------------------
    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        w = stats[
            i,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            i,
            cv2.CC_STAT_HEIGHT
        ]

        aspect = max(
            w,
            h
        ) / max(
            1,
            min(w, h)
        )

        # ----------------------------------------------
        # SAFETY CHECK #2
        # Huge bounding boxes are probably not text
        # ----------------------------------------------
        if (
            w > w_img * 0.40 or
            h > h_img * 0.20
        ):
            keep_mask[
                labels == i
            ] = 255
            continue

        # Likely text
        if (
            area < 1 and
            h < 50 and
            aspect < 10
        ):
            continue

        keep_mask[
            labels == i
        ] = 255

    # Morphological cleanup
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

    # White background
    result = np.full_like(
        img,
        255
    )

    result[
        keep_mask > 0
    ] = img[
        keep_mask > 0
    ]

    # --------------------------------------------------
    # SAFETY CHECK #3
    # Too much image removed
    # --------------------------------------------------
    removed_pixels = np.sum(
        keep_mask == 0
    )

    removed_ratio = (
        removed_pixels /
        total_pixels
    )

    print(
    f"Removed Ratio = {removed_ratio:.1%}"
    )


    if removed_ratio < min_removed_ratio:
        print(
            "[removeText] FAIL: "
            f"removed only {removed_ratio:.1%} "
            "of image"
        )
        return blank_page(img), 1


    # --------------------------------------------------
    # SAFETY CHECK #4
    # Too much image changed
    # --------------------------------------------------
    diff = cv2.absdiff(
        img,
        result
    )

    changed_pixels = np.sum(
        np.any(
            diff > 10,
            axis=2
        )
    )

    changed_ratio = (
        changed_pixels /
        total_pixels
    )

    if changed_ratio > max_changed_ratio:
        print(
            "[removeText] FAIL: "
            f"changed {changed_ratio:.1%} "
            "of image"
        )
        return blank_page(img), 1

    return result, 0