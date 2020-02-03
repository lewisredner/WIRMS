import numpy as np
import cv2 as cv
# pip install opencv-contrib-python==3.4.2.16 for sift and surf w/o headache
# had to add this weird stuff to make it work, don't know why. Originally just import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageDraw
import os
import math
from netCDF4 import Dataset


class ImageProcessing(object):

    # initialise all necessary variables
    def __init__(self, pixel_inch, grid_size, folder_name, file_name, base_name, save_name, save_folder, N_MATCHES = 500):
        # pixels/inch
        self.pixels_inch = pixel_inch
        # size of grid
        self.grid_size = grid_size
        # number of grids
        self.step_size = pixel_inch*grid_size
        # name of folder that fields are saved to
        self.folder_name = folder_name
        # name of image that needs to be SIFTed
        self.file_name = file_name
        # name of image that everything is compared to
        self.base_name = base_name
        # name of final saved image
        self.save_name = save_name
        # folder where everything is saved
        self.save_folder = save_folder

        # get current working directory
        current_cd = os.getcwd()
        # generate save location
        self.save_location = os.path.join(current_cd, self.save_folder)

        # check to see if the desired save folder exists, if not, then create it
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)

        # generate image save name
        self.final_save_name = os.path.join(self.save_location, self.save_name)

        # generate read name. I have the safety check in the function
        self.read_location = os.path.join(current_cd, self.folder_name)

        # optional for the number of matches
        self.N_MATCHES = N_MATCHES

    # reads in image and implements SIFT algorithm
    def SIFT(self):

        # assign image names
        # 1st image is baseline, second is bad image

        # look in the photos folder in the cd to extract the images from

        img1_name = os.path.join(self.read_location, self.base_name)
        img2_name = os.path.join(self.read_location, self.file_name)

        try:
            # read image and store in colour
            img_colour = cv.imread(img1_name, 1)
            img_colour2 = cv.imread(img2_name, 1)

            # read image and store in grayscale
            img = cv.imread(img1_name, cv.IMREAD_GRAYSCALE)
            img2 = cv.imread(img2_name, cv.IMREAD_GRAYSCALE)

            # raise an exception if we can't read the images
            if img is None or img2 is None:
                raise Exception('Folder "Photos" or file does not exist')

        # except case if can't read the image
        except Exception:
            print(
                'Folder "Photos" does not exist within your current directory or does not contain the filename provided')
            # if image isn't in right, then it's a waste of time to try SIFT
            exit()

        try:
            # create sift env
            sift = cv.xfeatures2d.SIFT_create()

            # get key points and descriptors
            kp, desc = sift.detectAndCompute(img, None)
            kp2, desc2 = sift.detectAndCompute(img2, None)

            # use bf for matches
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

            matches = bf.match(desc, desc2)

            # should be 500
            #N_MATCHES = 500

            # match image
            match_img = cv.drawMatches(
                img, kp,
                img2, kp2,
                matches[:self.N_MATCHES], img2.copy(), flags=0)

            plt.figure(figsize=(12, 6))

            # save figure with keypoint correlations
            kp_save_name = ImageProcessing.save_stitcher(self, '_keypoints.')
            # save image
            cv.imwrite(kp_save_name, match_img)

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

            # get current working directory
            current_cd = os.getcwd()

            save_location = os.path.join(current_cd, self.save_folder)

            # check to see if the desired save folder exists, if not, then create it
            if not os.path.exists(save_location):
                os.makedirs(save_location)

            # generate image save name
            final_save_name = os.path.join(save_location, self.save_name)
            # save image using the final save name generated
            cv.imwrite(final_save_name, im2Reg)
            plt.close()


        except:
            print('SIFT environment not accessible')

    # adds grids to image
    def image_gridding(self):

        location = os.getcwd()

        name = os.path.join(location, self.folder_name, self.base_name)

        # use base image for consistent grid sizes
        base = Image.open(name)
        # open final image and intitialise height and width
        im = Image.open(self.final_save_name)
        draw = ImageDraw.Draw(im)
        y_start = 0
        y_end = im.height
        # set step size using base width for consistency across images
        step_size = self.step_size
        # iterate through and draw lines vertically and then horizontally
        for x in range(0, im.width, step_size):
            line = ((x, y_start), (x, y_end))
            draw.line(line, fill=128)

        x_start = 0
        x_end = im.width
        # was im.height
        for y in range(0, im.height, step_size):
            line = ((x_start, y), (x_end, y))
            draw.line(line, fill=128)
        # don't know what this does but keep it
        del draw

        # im.show()

        grid_save_name = ImageProcessing.save_stitcher(self, '_grid.')

        im.save(grid_save_name)

    # improvement of grid values
    def grid_values(self):
        # read in image
        im = cv.imread(self.final_save_name, 0)

        # get image dimensions
        height, width = im.shape

        # set step size using base width for consistency across images
        step_size = self.step_size

        # initialise variables
        x_current = 0
        y_current = 0

        # calculate number of grids, and round up for partial vertical grids in order to count them
        gridsx = int(math.ceil(width/self.step_size))
        gridsy = int(math.ceil(height/self.step_size))

        # preallocate size of stdev and average for MAXIMUM SPEED
        stdev = np.zeros((gridsy, gridsx), dtype=np.float32)
        # mean
        average = np.zeros((gridsy, gridsx), dtype=np.float32)

        # ignoring all perfect black values, so convert image to float
        im_float = np.float32(im)
        # convert 0 values to nans
        im_float[im_float == 0] = np.nan
        # get standard deviation and mean of each grid
        for i in range(gridsy):
            # iterate through for the number of grids in the x direction
            for j in range(gridsx):
                # iterate through for the values in each grid along the x then y axes from bottom to top
                # (use nanstd/mean to ignore the nans, i.e. pure black)

                # if dealing with a partial grid, use height as the upper bound
                if y_current + step_size > height:
                    stdev[i, j] = np.nanstd(im[y_current:height, x_current:x_current + step_size])
                    average[i, j] = np.nanmean(im[y_current:height, x_current:x_current + step_size])
                # if looking at a whole grid, use the full grid (i.e. the whole step)
                else:
                    stdev[i, j] = np.nanstd(im[y_current:y_current + step_size, x_current:x_current + step_size])
                    average[i, j] = np.nanmean(im[y_current:y_current + step_size, x_current:x_current + step_size])
                # iterate the x value across the width of the image
                x_current = x_current + step_size
            # reset the x location
            x_current = 0
            # iterate the y location to end
            y_current = y_current + step_size
        # return the arrays to the function call
        return stdev, average

    # internal function to generate file save names
    def save_stitcher(self, addition, h5=0):
        # Use regular expressions to search the save name until the punctuation (.) to get the filename
        new_name = re.compile(r"\w+")
        new_word = new_name.findall(self.save_name)
        # if we want to create the data file, parse in a non zero karg to change the file extension
        if h5 != 0:
            out_name = new_word[0] + addition + 'nc'
        else:
            # add addition to file name and save image w/ grids
            out_name = new_word[0] + addition + new_word[1]
        # recreate directory and save
        out_save_name = os.path.join(self.save_location, out_name)
        return out_save_name

    # writes numerical pixel values to a hdf5 file
    def write_to_file(self, std, ave):
        # generate save name
        net_save_name = ImageProcessing.save_stitcher(self, '_statistics.', 1)
        # create root group
        rootgrp = Dataset(net_save_name, "w", format = "NETCDF4")
        # create sub groups
        meangrp = rootgrp.createGroup("Mean")
        stdgrp = rootgrp.createGroup("Standard Deviation")
        # create dimensions of groups
        kms = ave.shape
        meangrp.createDimension('average', ave.shape)
        stdgrp.createDimension('std', std.shape)
        # create variables
        averages = meangrp.createVariable("average","f8",("average"))
        stds = stdgrp.createVariable("std","f8",("standard deviation"))
        # assign data
        averages[:] = ave
        stds[:] = std


# iterate through every file in the folder photos_test
i = 0
for filename in os.listdir(os.path.join(os.getcwd(), "nir_test")):
    # inputs: pixels/inch, size of each grid in inches, filename, baseline image name, output name, output folder
    imPr = ImageProcessing(10, 7, "nir_test", filename, "nir_test_0.PNG", "Final" + str(i) + ".png", "NIR Test Out")

    imPr.SIFT()

    imPr.image_gridding()

    [standard_dev, average] = imPr.grid_values()

    imPr.write_to_file(standard_dev, average)

    i = i+1



