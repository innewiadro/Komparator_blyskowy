import os
from pathlib import Path
import numpy as np
import cv2 as cv

MIN_NUM_KEYPOINT_MATCHES = 50


def find_best_matches(img1, img2):
    """returns a list of key points and a list of best matches"""
    orb = cv.ORB_create(nfeatures=100)  # create object ORB
    kp1, desc1 = orb.detectAndCompute(img1, mask=None)
    kp2, desc2 = orb.detectAndCompute(img2, mask=None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossChecker=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    best_matches = matches[:MIN_NUM_KEYPOINT_MATCHES]
    return kp1, kp2, best_matches


def qc_best_matches(img_match):
    """draw matches of points connected by lines"""
    cv.imshow(f"{MIN_NUM_KEYPOINT_MATCHES} najlepiej dopasowane punkty kluczowe", img_match)  # TODO english
    cv.waitKey(3000)


def register_image(img1, img2, kp1, kp2, best_matches):
    """returns the first image recorded to the second"""
    if len(best_matches) >= MIN_NUM_KEYPOINT_MATCHES:
        src_pts = np.zeros((len(best_matches), 2), dtype=np.float32)
        dst_pts = np.zeros((len(best_matches), 2), dtype=np.float32)
        for i, match in enumerate(best_matches):
            src_pts[i, :] = kp1[match.queryIdx].pt
            dst_pts[i, :] = kp2[match.trainIdx].pt
        h_array, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)
        height, width = img2.shape
        img1_wraped = cv.wrapPerspective(img1, h_array, (width, height))
        return img1_wraped
    else:
        print(f"WARNING: Number of key points less than {MIN_NUM_KEYPOINT_MATCHES}\n")  # TODO eglish
        return img1


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
        qc_best_matches(img_match)
        img1_registred = register_image(img1, img2, kp1, kp2, best_matches)

        blink(img1, img1_registred, "verification of registration", num_loops=5)
        out_filename = f"{night1_files[i][:-4]}_registered.png"

        cv.imwrtie(str(path3 / out_filename), img1_registred)
        cv.destroyAllWindows()
        blink(img1_registred, img2, "Blink Comparator", num_loops=15)
