import math
import numpy as np
import cv2
from math import floor
from matplotlib import pyplot as plt
import time
from multiprocessing import Pool
import sys

class HybridImage:
    '''
    This class is used for creating hybrid images as well as 
    displaying a high-pass or low-pass version of an image
    '''
    def __init__(self, image1, output_filename, image2=None, sigma=4.0):
        '''the initialization of the class

        :param str image1: the location of the first image;
        :param str output_filename: set the filename of the output;
        :param str image2: the location of the second image;
        :param float sigma: set the sigma, its default value is 4.0.

        '''
        self.image1 = cv2.imread(image1)
        if image2:
            # if I just want to output the high-pass or low-pass version of an image,
            # the parameter of image2 won't be used,
            # so I explicitly set its default value to None.
            self.image2 = cv2.imread(image2)
        self.filename = output_filename
        self.sigma = float(sigma)
 
    def gaussian_template(self):
        # compute the window size according to the method specified in the coursework detail
        winsize = int(8.0 * self.sigma + 1.0)
        if winsize % 2 == 0:
            winsize += 1   # the window size must be odd
        # initialize a 2-D array for the template and set the values to zeros
        template = np.zeros((winsize,winsize))
        # compute the centre of the template
        centre = math.floor(winsize/2) + 1
        # create a gaussian template
        sum = 0
        for i in range(1, winsize + 1):
            for j in range(1, winsize + 1): 
                template[j - 1][i - 1] = math.exp(-((i-centre) * (i-centre) + (j-centre) * (j-centre))/(2 * self.sigma * self.sigma))
                sum += template[j - 1][i - 1]
        return template/sum  # normalise by the total sum

    def template_convolution(self, image, template):
        '''
        the function for template convolution

        :param array image: a 2-D matrix of an image
        :param array template: a 2-D matrix of a template to be convolved with the image

        '''
        # get the shapes of the image and the template
        rows, cols = image.shape
        trows, tcols = template.shape
        # filp the template
        flipped_template = np.zeros((trows, tcols))
        for i in range(trows):
            for j in range(tcols):
                flipped_template[i][j] = template[trows - i - 1][tcols - j - 1]
        # get the half of the rows and coloumns of the template
        tr_half = floor(trows / 2)
        tc_half = floor(tcols / 2)
        # initialize the resulting image to black(zero brightness levels)
        output_image = np.zeros(image.shape)
        # convolve the template
        for x in range(tc_half, cols - tc_half):  # address all columns except border
            for y in range(tr_half, rows - tr_half): # address all rows except border
                # get the area of the image which corresponds to the location of the template
                pixel_set = image[y - tr_half:y + tr_half + 1, x - tc_half: x + tc_half + 1]
                # get the resulting value of the point by multiplying two matrices element-wise and getting the sum.
                output_image[y][x] = floor(np.multiply(flipped_template, pixel_set).sum()) 
        return output_image

    def low_pass(self, image):
        '''
        create the low-pass version of an image

        :param array image: a 2-D matrix of an image

        '''
        # create a gaussian template
        template = self.gaussian_template()
        # split the image into 3 separate color channels
        b, g, r = cv2.split(image)
        # create a pool with 3 processes
        pool = Pool(processes=3)
        # assign a task of template convolution to an independent process
        # and run the processes in an asynchronous way
        multi_res = [pool.apply_async(self.template_convolution, (i, template)) for i in (b, g, r)]
        # get the result from the processes
        res = [i.get() for i in multi_res]
        # merge three color channels into a new image
        new_image = np.uint8(cv2.merge(res))
        return new_image

    def high_pass(self, low_pass, original):
        '''
        create the high-pass version of an image

        :param array low_pass: a low-pass version of an image
        :param array original: the original image

        '''
        # get the high-pass version of an image 
        # by subtracting the low-pass version from the image itself
        return original - low_pass

    def ouput_lowpass(self):
        '''
        get the low-pass version of an image and output it
        '''
        low = self.low_pass(self.image1)
        cv2.imwrite(self.filename, low)
        
    def ouput_highpass(self):
        '''
        get the high-pass version of an image and output it
        '''
        low = self.low_pass(self.image1)
        # The high frequency image can be visualised by adding 0.5 to every pixel in each colour channel.
        high = self.high_pass(low, self.image1) + 128
        cv2.imwrite(self.filename, high)

    def main(self):
        '''
        create a hybrid image
        '''
        # mark the start time of the operation
        start = time.time()
        # get the low-pass versions of the two input images
        low_pass1 = self.low_pass(self.image1)
        low_pass2 = self.low_pass(self.image2)
        # get the high-pass version of the first input image
        high_pass1 = self.high_pass(low_pass1, self.image1)
        # create a hybrid image
        # using the high-pass version of the first input image
        # and the low-pass version of the second input image
        hybrid_img = high_pass1 + low_pass2
        # output the execution time of the operations above
        print('Execution time: %.2f s' % (time.time() - start))
        # plot the two original images and their hybrid image
        plt.subplot(131), plt.imshow(cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)), plt.title('image1')
        plt.subplot(132), plt.imshow(cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)), plt.title('image2')
        plt.subplot(133), plt.imshow(cv2.cvtColor(hybrid_img, cv2.COLOR_BGR2RGB)), plt.title('hybrid')
        plt.show()
        # output the hybrid image
        cv2.imwrite(self.filename, hybrid_img)


if __name__ == '__main__':
    if sys.argv[1] in ['-ol', '-oh']:  # used for outputting the low-pass or high-pass version of an image
        image = sys.argv[2]
        filename = sys.argv[3]
        sigma = sys.argv[-1] if len(sys.argv) > 4 else 4.0
        hi = HybridImage(image1=image, output_filename=filename, sigma=sigma)
        if sys.argv[1] == '-ol':
            hi.ouput_lowpass()
        else:
            hi.ouput_highpass()
    else:  # used for creating a hybrid image from two images
        image1 = sys.argv[1]
        image2 = sys.argv[2]
        output_filename = sys.argv[3]
        sigma = sys.argv[-1] if len(sys.argv) > 4 else 4.0
        hybrid_img = HybridImage(image1, output_filename, image2, sigma)
        hybrid_img.main()