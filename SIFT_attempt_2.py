import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageDraw

# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16 for sift and surf w/o headache


def grid_values(name, step_size, stepcount):
    # read in image (colour)
    im = cv.imread(name, 0)

    # get image dimensions (will break with grayscale image)
    height, width = im.shape

    # initialise variables
    x_current = 0
    y_current = 0

    gridsx = int(width/step_size)
    gridsy = int(height/step_size)
    # preset array sized for the number of grids horizontally and vertically
        # std
    stdev = np.zeros((gridsy, gridsx), dtype=np.float32)
        # mean
    average = np.zeros((gridsy, gridsx), dtype=np.float32)
    # get standard deviation of each grid
    for i in range(gridsy):
        # iterate through for the number of grids in the x direction
        for j in range(gridsx):
            # iterate through for the values in each grid along the x then y axes from bottom to top
            stdev[i, j] = np.std(im[x_current:x_current+step_size, y_current:y_current+step_size])
            average[i, j] = np.mean(im[x_current:x_current+step_size, y_current:y_current+step_size])
            # iterate the x value across the width of the image
            x_current = x_current + step_size
        # reset the x location
        x_current = 0
        # iterate the y location to end
        y_current = y_current + step_size
    return stdev, average


#for x in range(step_size):
#    # take the standard deviation for every column in the grid, limited by the y constraints imposed
#    temp[x] = np.std(im[x_current + x, y_current:y_current + step_size])
# take the standard deviation of all the standard deviations in the grid
#a[i, j] = np.std(temp)

############################################################################
def imageGridding(base_name, name, stepcount):
    #use base image for consistent grid sizes
    base = Image.open(base_name)
    # open final image and intitialise height and width
    im = Image.open(name)
    draw = ImageDraw.Draw(im)
    y_start = 0
    y_end = im.height
    # set step size using base width for consistency across images
    step_size = int(base.width/stepcount)
    # iterate through and draw lines vertically and then horizontally
    for x in range(0, im.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = im.width
    #was im.height
    for y in range(0, im.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)
    # don't know what this does but keep it
    del draw

    #im.show()

    # Use regular expressions to search the save name until the punctuation (.) to get the filename
    new_name = re.compile(r"\w+")
    new_word = new_name.findall(name)
    # add grid to file name and save image w/ grids
    grid_name = new_word[0] + '_grid.' + new_word[1]

    im.save(grid_name)

    # get details about each grid
    grid_values(name, step_size, stepcount)

############################################################################

    # I have a folder for images because they suck
path = r'C:\Users\Lewis\Desktop\Senior_Projects\Photos'

# assign image names
    # 1st image is baseline, second is bad image
img1_name = path+'\\'+'whats1.jpeg'
img2_name = path+'\\'+'whats3.jpeg'

try:
    # read image and store in colour
    img_colour = cv.imread(img1_name, 1)
    img_colour2 = cv.imread(img2_name, 1)

    #cv.imwrite('colour rotated2.jpg',img_colour2)

    # read image and store in gray scale
    img = cv.imread(img1_name, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2_name, cv.IMREAD_GRAYSCALE)

    #cv.imwrite('grayscale.png', img2)

    # create sift env
    sift = cv.xfeatures2d.SIFT_create()

    # get key points and descriptors
    kp, desc = sift.detectAndCompute(img, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    # use bf for matches
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    matches = bf.match(desc, desc2)

    # should be 500
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

    # save image
    imSaveName = 'Final.png'
    cv.imwrite(imSaveName, im2Reg)


except:
    print('That''s not a bingo!')

# add grids
imageGridding(img1_name, imSaveName, 10)
