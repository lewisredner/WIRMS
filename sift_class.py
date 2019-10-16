import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16 for sift and surf w/o headache

class image_processing(object):

    def __init__(self,ideal_image_fn, reduction_meth = 'mean'):
        """
        
        Parameters:

        ideal_image_fn : string of file directory of the ideal image 

        reduction_meth : method of data reduction onto grid points 

        """
        #check if file exists 

        #read in file and store color 
        self.img_colour = cv.imread(ideal_image_fn, 1)

        #read image as gray scale 
        self.img = cv.imread(ideal_image_fn, 0)

        # note, shape outputs colour channels too iff its a colour image
        self.height, self.width = self.img.shape

        #declare sift environment 
        self.sift = cv.xfeatures2d.SIFT_create()

        #get key points of the ideal image 
        self.kp, desc = sift.detectAndCompute(self.img, None)


        #set up grid point stufff, 

    def __call__(self, actual_image_fn, N_MATCHES = 500, save_file = True):
        """
        actual_image_fn : string of file directory of the image that we want to match with the ideal image 
        
        """

        img_colour2 = cv.imread(img2_name,1)

        img2 = cv.imread(img2_name, 0)

        im2Reg = self.SIFT_image_alignment(img2, save_file)

        

        #gridpoint stuff 

    def SIFT_image_alignment(self, img2):
        """
        

        """
        sift = self.sift
        
        kp2, desc2 = sift.detectAndCompute(img2, None)

        # use bf for matches
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(self.desc, desc2)

        match_img = cv.drawMatches(
            self.img, self.kp,
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
        # assign to bad image
        im2Reg = cv.warpPerspective(img_colour2, h, (self.width, self.height))

        if (save_file):
            cv.imwrite('Final.jpg', im2Reg)


        return im2Reg

    def grid_point(self, image):
        """
        This function will reduce a given image into geogrpahic grid points 

        Parameters:

        """

        return 1 

    def grid_mean(self, image):

        """
        This function 

        Paramters:



        """
        return 1 



# assign image names
img1_name = 'field_trans.jpg'
img2_name = 'field_tr2.jpg'

# assign image names
img1_name = 'test1.jpg'
img2_name = 'test2.jpg'



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
