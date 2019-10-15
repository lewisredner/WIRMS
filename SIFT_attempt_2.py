import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16 for sift and surf w/o headache

# assign image names
    # 1st image is baseline, second is bad image
img1_name = 'reelee.jpg'
img2_name = 'reelee2.jpg'

# read image and store in colour
img_colour = cv.imread(img1_name,1)
img_colour2 = cv.imread(img2_name,1)

#cv.imwrite('colour rotated2.jpg',img_colour2)

# read image and store in gray scale
img = cv.imread(img1_name, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(img2_name, cv.IMREAD_GRAYSCALE)

# create sift env
sift = cv.xfeatures2d.SIFT_create()

# get key points and descriptors
kp, desc = sift.detectAndCompute(img, None)
kp2, desc2 = sift.detectAndCompute(img2, None)

# use bf for matches
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

matches = bf.match(desc, desc2)

N_MATCHES = 500

# match image
match_img = cv.drawMatches(
    img, kp,
    img2, kp2,
    matches[:N_MATCHES], img2.copy(), flags=0)




plt.figure(figsize=(12,6))

cv.imwrite('Keypoint_alignment.png', match_img)


# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# Find homography
# put pointsx where the first entry is the rotated one
h, mask = cv.findHomography(points2, points1, cv.RANSAC)

# Use homography
# extract from good image
height, width = img.shape
# assign to bad image
im2Reg = cv.warpPerspective(img_colour2, h, (width, height))

# save to image
cv.imwrite('Final.jpg', im2Reg)
