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
    img2 = io.imread(img1_dir)

    extract_func = "ORB"

    # Extract key points and descriptors
    # Three types of feature extraction
    # 1. SURF
    # 2. ORB
    # 3. SIFT
    kp1, des1 = ExtractFeatures(img1, extract_func, num_features)
    kp2, des2 = ExtractFeatures(img2, extract_func, num_features)

    print(img1.shape)
    print(des1.shape)

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw all matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

    # Display results
    fig = plt.figure()
    plt.imshow(img3)
    plt.axis('off')
    plt.show()