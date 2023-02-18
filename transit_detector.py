import os
from pathlib import Path
import cv2 as cv

#  width of the area to be ignored counted in pixels from the edge
PAD = 5


def find_transit(image, diff_image, pad):
    """search and outlines the phenomena visible against the background of a fragment of the sky"""
    transient = False
    height, width = diff_image.shape
    cv.rectangle(image, (PAD, PAD), (width - PAD, height - PAD), 255, 1)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(diff_image)
    if pad < max_loc[0] < width - pad and pad < height - pad:
        cv.circle(image, max_loc, 10, 255, 0)
        transient = True
    return transient, max_loc


def main():
    """creation of a list of directory contents and definition of paths"""
    night1_files = sorted(os.listdir("night_1_registered_transients"))
    print(night1_files)
    night2_files = sorted(os.listdir("night_2"))
    path1 = Path.cwd() / "night_1_registered_transients"

    path2 = Path.cwd() / "night_2"

    path3 = Path.cwd() / "night_1_2_transients"

    """loop through images"""

    for i, _ in enumerate(night1_files):

        img1 = cv.imread(str(path1 / night1_files[i]), cv.IMREAD_GRAYSCALE)
        print(img1)

        img2 = cv.imread(str(path2 / night2_files[i]), cv.IMREAD_GRAYSCALE)
        print(img2)
        diff_imgs1_2 = cv.absdiff(img1, img2)
        cv.imshow("difference", diff_imgs1_2)
        cv.waitKey(2000)

        temp = diff_imgs1_2.copy()
        transient1, transient_loc1 = find_transit(img1, temp, PAD)
        cv.circle(temp, transient_loc1, 10, 0, -1)

        transient2, _ = find_transit(img1, temp, PAD)

        """outlines object and save results"""
        if transient1 or transient2:
            print(f"Detected on {night1_files[i]} and {night2_files[i]}")
            font = cv.FONT_HERSHEY_COMPLEX_SMALL
            cv.putText(img1, night1_files[i], (10, 25), font, 1, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(img1, night2_files[i], (10, 55), font, 1, (255, 255, 255), 1, cv.LINE_AA)

            blended = cv.addWeighted(img1, 1, diff_imgs1_2, 1, 0)
            cv.imshow("Checked", blended)
            cv.waitKey(2500)

            out_filename = f'{night1_files[i][:-4]}_DETECTED.PNG'
            cv.imwrite(str(path3 / out_filename), blended)

        else:
            print(f"\nNot Detected on images {night1_files[i]} {night2_files[i]}")


if __name__ == "__main__":
    main()



