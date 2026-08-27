import cv2
import numpy as np
import sys

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

    if flag:
        combined = cv2.bitwise_not(blackWhite.copy())
    else:
        combined = green_mask.copy()

    combined[walls < 128] = 0

    cv2.imwrite(
        "../output/combined_Step6.png",
        combined
    )

    # np.save("../output/combined_Step6.npy", combined)

    print("Done")

    # return output_file