import cv2
import numpy as np
import sys
from pathlib import Path
import json

from keep_Green import keepGreen
from solid_Green import solidGreen
from remove_Green import removeGreen
from remove_Text import removeText
from extract_Walls import extractWalls

# used as 
# git submodule add <visual-repo-url> visual

# from combine import combineThem

# def process_floorplan(input_file, output_file):

input_file = sys.argv[1]

img = cv2.imread(input_file)

# run keep_Green keepGreen
green_only = keepGreen(img)

# print (type(green_only))

# cv2.imwrite(
#     "../output/green_only_Step1.png",
#     green_only
# )

# exit()
# branch off extract green
# run solid_Green solidGreen

solid_green, green_mask = solidGreen(
    green_only
)

# print (type(solid_green))

# cv2.imwrite(
#     "../output/solid_green_Step2.png",
#     solid_green
# )

# cv2.imwrite(
#     "../output/green_mask_Step2.png",
#     green_mask
# )

# run remove_Green
blackWhite = removeGreen(green_only)

# cv2.imwrite(
#     "../output/blackWhite_Step3.png",
#     blackWhite
# )

nonText, flag = removeText(blackWhite)

# cv2.imwrite(
#     "../output/nonText_Step4.png",
#     nonText
# )




gray, thresh, walls = extractWalls(nonText)

# cv2.imwrite(
#     "../output/gray_Step5.png",
#     gray
# )

# cv2.imwrite(
#     "../output/thresh_Step5.png",
#     thresh
# )

# cv2.imwrite(
#     "../output/walls_Step5.png",
#     walls
# )

# if flag:
#     combined = cv2.bitwise_not(blackWhite.copy())
# else:
#     combined = green_mask.copy()

# combined[walls < 128] = 0


# # output_file = (
# #     Path(__file__).resolve().parent.parent
# #     / "output"
# #     / "combined_Step6.png"
# # )

# output_file = sys.argv[2]

# print("Saving to:", output_file)

# cv2.imwrite(
#     str(output_file),
#     combined
# )


# # np.save("../output/combined_Step6.npy", combined)

# print("Done")

#     # return output_file

if flag:
    combined = cv2.bitwise_not(blackWhite.copy())
else:
    combined = green_mask.copy()

combined[walls < 128] = 0



if flag:
    combined = cv2.bitwise_not(blackWhite.copy())
else:
    combined = green_mask.copy()


def create_white_mask(image):
    """
    Create a binary mask where white pixels are 255 and
    non-white pixels are 0.
    """
    if image.ndim == 3:
        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray_image = image.copy()

    _, binary_mask = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary_mask


# ---------------------------------------------------------
# Measure the white area before applying the wall mask.
# ---------------------------------------------------------
white_mask_before = create_white_mask(combined)
white_before_wall_mask = cv2.countNonZero(
    white_mask_before
)


# In extractWalls output, values below 128 represent pixels
# that the current processing considers part of the wall mask.
wall_mask = walls < 128

wall_mask_pixels = int(
    np.count_nonzero(wall_mask)
)


# ---------------------------------------------------------
# Apply the wall mask.
# ---------------------------------------------------------
combined[wall_mask] = 0


# ---------------------------------------------------------
# Measure the white area after applying the wall mask.
# ---------------------------------------------------------
white_mask_after = create_white_mask(combined)
white_after_wall_mask = cv2.countNonZero(
    white_mask_after
)

white_removed_by_wall_mask = max(
    0,
    white_before_wall_mask - white_after_wall_mask
)

if white_before_wall_mask > 0:
    white_removed_ratio = (
        white_removed_by_wall_mask
        / white_before_wall_mask
    )
    white_retained_ratio = (
        white_after_wall_mask
        / white_before_wall_mask
    )
else:
    white_removed_ratio = 0.0
    white_retained_ratio = 0.0


# ---------------------------------------------------------
# Save processed output.
# ---------------------------------------------------------
output_file = Path(sys.argv[2])

print("Saving to:", output_file)

success = cv2.imwrite(
    str(output_file),
    combined
)

if not success:
    raise RuntimeError(
        f"Unable to save processed image: {output_file}"
    )


# ---------------------------------------------------------
# Save diagnostic masks.
#
# These make it easy to visually inspect which pixels were
# removed by the wall operation.
# ---------------------------------------------------------
debug_before_file = output_file.with_name(
    f"{output_file.stem}_white_before.png"
)

debug_after_file = output_file.with_name(
    f"{output_file.stem}_white_after.png"
)

debug_wall_mask_file = output_file.with_name(
    f"{output_file.stem}_wall_mask.png"
)

cv2.imwrite(
    str(debug_before_file),
    white_mask_before
)

cv2.imwrite(
    str(debug_after_file),
    white_mask_after
)

cv2.imwrite(
    str(debug_wall_mask_file),
    np.where(
        wall_mask,
        255,
        0
    ).astype(np.uint8)
)


# ---------------------------------------------------------
# Save processing statistics in a sidecar JSON file.
#
# Example:
# processed_sess_123.png
# processed_sess_123.stats.json
# ---------------------------------------------------------
stats_file = output_file.with_suffix(
    ".stats.json"
)

processing_stats = {
    "image_width_pixels": int(combined.shape[1]),
    "image_height_pixels": int(combined.shape[0]),
    "white_pixels_before_wall_mask": int(
        white_before_wall_mask
    ),
    "white_pixels_after_wall_mask": int(
        white_after_wall_mask
    ),
    "white_pixels_removed_by_wall_mask": int(
        white_removed_by_wall_mask
    ),
    "wall_mask_pixels": wall_mask_pixels,
    "white_removed_ratio": float(
        white_removed_ratio
    ),
    "white_retained_ratio": float(
        white_retained_ratio
    )
}

with open(
    stats_file,
    "w",
    encoding="utf-8"
) as stats_handle:
    json.dump(
        processing_stats,
        stats_handle,
        indent=4
    )


print("")
print("=" * 60)
print("[WALL MASK AREA DIAGNOSTICS]")
print(
    f"Image dimensions: "
    f"{combined.shape[1]} x {combined.shape[0]} px"
)
print(
    f"White pixels before wall mask: "
    f"{white_before_wall_mask:,}"
)
print(
    f"White pixels after wall mask: "
    f"{white_after_wall_mask:,}"
)
print(
    f"White pixels removed: "
    f"{white_removed_by_wall_mask:,}"
)
print(
    f"White removed ratio: "
    f"{white_removed_ratio:.2%}"
)
print(
    f"White retained ratio: "
    f"{white_retained_ratio:.2%}"
)
print(
    f"Statistics saved to: {stats_file}"
)
print("=" * 60)
print("")
print("Done")