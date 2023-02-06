import os
from pathlib import Path
import numpy as np
import cv2 as cv

MIN_NUM_KEYPOINT_MATCHES = 50


def find_best_matches():
    """returns a list of key points and a list of best matches"""
    pass


def qc_best_matches():
    """draw matches of points connected by lines"""
    pass


def register_image():
    """returns the first image recorded to the second"""
    pass


def blink():
    """display two images"""
    pass


def main():
    """ creates a loop through the contents of two folders, registers the images and displays them alternately"""
    night1_files = sorted(os.listdir("night_1"))
    night2_files = sorted(os.listdir("night_2"))
    path1 = Path.cwd() / "night_1"
    path2 = Path.cwd() / "night_2"
    path3 = Path.cwd() / "night_1_registred"

    # main loop

    for i, _ in enumerate(night1_files):
        img1 = cv.imread(str(path1 / night1_files[i]), cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(str(path2 / night2_files[i]), cv.IMREAD_GREYSCALE)
        print(f"comparison {night1_files[i]} with {night2_files[i]}")
        kp1, kp2, best_matches = find_best_matches(img1, img2)
        img_match = cv.drawMatches(img1, kp1, img2, kp2, best_matches)

        height, width = img1.shape
        cv.line(img_match, (width, 0), (width, height), (255, 255, 255), 1)
        # qc_best_matches(img_match)
        img1_registred = register_image(img1, img2, kp1, kp2, best_matches)

        blink(img1, img1_registred, "verification of registration", num_loops=5)
        out_filename = f"{night1_files[i][:-4]}_registered.png"

        cv.imwrtie(str(path3 / out_filename), img1_registred)
        cv.destroyAllWindows()
        blink(img1_registred, img2, "Blink Comparator", num_loops=15)
