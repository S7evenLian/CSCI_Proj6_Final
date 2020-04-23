import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage import io 



def ExtractFeatures(img, extract_func, num_features):

    # extract features
    if extract_func is 'SIFT':
        sift = cv2.xfeatures2d.SIFT_create(num_features)
        keypoints, descriptors = sift.detectAndCompute(img, None)

    # elif extract_func is "SURF":
    #     surf = cv2.xfeatures2d.SURF_create(num_features)
    #     keypoints, descriptors = surf.detectAndCompute(img, None)

    # elif extract_func is "ORB":
    #     orb = cv2.ORB_create(num_features)
    #     keypoints, descriptors = orb.detectAndCompute(img, None)
    else:
        sys.exit("Please choose a valid func: SIFT, SURF, ORB")

    return keypoints, descriptors


if __name__ == '__main__':

    num_features = 200
    img1_dir = '../data/panorama-data1/DSC01538.JPG'
    img2_dir = '../data/panorama-data1/DSC01540.JPG'

    img1 = io.imread(img1_dir)
    img2 = io.imread(img2_dir)

    extract_func = "SIFT"

    # Extract key points and descriptors with SIFT
    # 1. SIFT only for now
    kp1, des1 = ExtractFeatures(img1, extract_func, num_features)
    kp2, des2 = ExtractFeatures(img2, extract_func, num_features)

    # source code from https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)

    # convert the matched key points to xy coordinates
    src_xy_coord = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
    dst_xy_coord = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
