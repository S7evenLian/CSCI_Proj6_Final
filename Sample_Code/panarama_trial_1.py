import os
import cv2
import imutils

img_dir = '../data/panorama-data1'
names = os.listdir(img_dir)
print(names)
images = []
for name in names:
    img_path = os.path.join(img_dir, name)
    image = cv2.imread(img_path)
    images.append(image)

stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)

if status==0:
    cv2.imwrite('stitch.jpg', stitched)