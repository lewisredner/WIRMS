import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageDraw
import os
import h5py

class ImageProcessing(object):

    def __init__(self, stepcount, folder_name, file_name, base_name, save_name, save_folder, N_MATCHES = 500):
        # steps for grid
        self.stepcount = stepcount
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

            kp_save_name = ImageProcessing.save_stitcher(self, '_keypoints.')

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

            cv.imwrite(final_save_name, im2Reg)


        except:
            print('That''s not a bingo!')
    # adds grids to image
    def imageGridding(self):

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
        step_size = int(base.width / self.stepcount)
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
    # takes pixel values of each grid OBSOLETE
    def grid_values(self):
        # read in image (colour)
        im = cv.imread(self.final_save_name, 0)

        # get image dimensions (will break with colour image)
        height, width = im.shape

        # set step size using base width for consistency across images
        step_size = int(width / self.stepcount)

        # initialise variables
        x_current = 0
        y_current = 0

        gridsx = int(self.stepcount)
        gridsy = int(height * self.stepcount / width)
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
                stdev[i, j] = np.std(im[y_current:y_current + step_size, x_current:x_current + step_size])
                average[i, j] = np.mean(im[y_current:y_current + step_size, x_current:x_current + step_size])
                # iterate the x value across the width of the image
                x_current = x_current + step_size
            # reset the x location
            x_current = 0
            # iterate the y location to end
            y_current = y_current + step_size
        return stdev, average
    # improvement of grid values
    def grid_values_test(self):
        # read in image
        im = cv.imread(self.final_save_name, 0)

        # get image dimensions
        height, width = im.shape

        # set step size using base width for consistency across images
        step_size = int(width / self.stepcount)

        # initialise variables
        x_current = 0
        y_current = 0

        gridsx = int(self.stepcount)
        gridsy = int(height * self.stepcount / width)
        # preset array sized for the number of grids horizontally and vertically

        # preallocate size of stdev and average for MAXIMUM SPEED
        stdev = np.zeros((gridsy, gridsx), dtype=np.float32)
        # mean
        average = np.zeros((gridsy, gridsx), dtype=np.float32)

        # ignoring all perfect black values, so convert image to float
        im_float = np.float32(im)
        # convert 0 values to nans
        im_float[im_float == 0] = np.nan
        # get standard deviation of each grid
        for i in range(gridsy):
            # iterate through for the number of grids in the x direction
            for j in range(gridsx):
                # iterate through for the values in each grid along the x then y axes from bottom to top
                # (use nanstd/mean to ignore the nans, i.e. pure black)
                stdev[i, j] = np.nanstd(im[y_current:y_current + step_size, x_current:x_current + step_size])
                average[i, j] = np.nanmean(im[y_current:y_current + step_size, x_current:x_current + step_size])
                # iterate the x value across the width of the image
                x_current = x_current + step_size
            # reset the x location
            x_current = 0
            # iterate the y location to end
            y_current = y_current + step_size
        return stdev, average
    # internal function to generate file save names
    def save_stitcher(self, addition, h5=0):
        # Use regular expressions to search the save name until the punctuation (.) to get the filename
        new_name = re.compile(r"\w+")
        new_word = new_name.findall(self.save_name)
        # if we want to create the data file, parse in a non zero karg to change the file extension
        if h5 != 0:
            out_name = new_word[0] + addition + 'hdf5'
        else:
            # add addition to file name and save image w/ grids
            out_name = new_word[0] + addition + new_word[1]
        # recreate directory and save
        out_save_name = os.path.join(self.save_location, out_name)
        return out_save_name
    # writes numerical pixel values to a hdf5 file
    def write_to_file(self, std, ave):
        # generate a save name using save stitcher
        h5_save_name = ImageProcessing.save_stitcher(self, '_numerical.', 1)
        # create the h5py file
        f = h5py.File(h5_save_name, "w")
        # generate the two data sets and save the standard deviation and average to each set
        dset = f.create_dataset("standard deviation", std.shape, dtype='i')
        dset[...] = std

        dset2 = f.create_dataset("average", ave.shape, dtype='i8')
        dset2[...] = ave

imPr = ImageProcessing(50, "Photos", "whats8.jpeg", "whats1.jpeg", "Final.png", "Processed_Images")

imPr.SIFT()

imPr.imageGridding()

[standard_dev, average] = imPr.grid_values_test()

imPr.write_to_file(standard_dev, average)
