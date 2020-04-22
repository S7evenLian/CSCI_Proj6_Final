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

    elif extract_func is "SURF":
        surf = cv2.xfeatures2d.SURF_create(num_features)
        keypoints, descriptors = surf.detectAndCompute(img, None)

    elif extract_func is "ORB":
        orb = cv2.ORB_create(num_features)
        keypoints, descriptors = orb.detectAndCompute(img, None)
    else:
        sys.exit("Please choose a valid func: SIFT, SURF, ORB")

    return keypoints, descriptors

    
    



if __name__ == '__main__':

    num_features = 10
    img1_dir = '../data/panorama-data1/DSC01538.JPG'
    img2_dir = '../data/panorama-data1/DSC01540.JPG'

    img1 = io.imread(img1_dir)
    img2 = io.imread(img2_dir)

    extract_func = "SIFT"

    # Extract key points and descriptors
    # Three types of feature extraction
    # 1. SURF
    # 2. ORB
    # 3. SIFT
    kp1, des1 = ExtractFeatures(img1, extract_func, num_features)
    kp2, des2 = ExtractFeatures(img2, extract_func, num_features)

    # convert the keypoints to coordinates
    xy_kp1 = cv2.KeyPoint_convert(kp1)
    xy_kp2 = cv2.KeyPoint_convert(kp2)

    print(xy_kp1)
    print(xy_kp2)
    
