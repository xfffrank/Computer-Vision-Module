import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

def visualization():
    '''
    Visualize the effect of a hybrid image by progressively down-sampling it.
    '''
    # get the hybrid image from the command
    filename = sys.argv[1]
    # load the image
    image = cv2.imread(filename)
    # create different scales of the image by setting the range of 0.1 to 1.2 with an interval of 0.3
    resized = [cv2.resize(image, (0,0), None, i, i) for i in np.arange(0.1, 1.2, .3)]
    # reverse the order so that the smallest image is at the end
    resized.reverse()
    num_img = len(resized)
    # set the size of the figure
    fig = plt.figure(figsize=[10, 2])
    # adjust the spacing between subplots
    plt.subplots_adjust(hspace=0, wspace=0.1, left=0, right=1, bottom=0, top=1)
    for index, image in enumerate(resized, start=1):
        if index == 1:
            # plot the original image,
            # and get the limits of its x axis and y axis,
            # which can be shared with the rest of the images.
            ax = fig.add_subplot(1, num_img, 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # hide axis
            bottom, top = ax.get_ylim()
            left, right = ax.get_xlim()
            ax.set_ylim([bottom, top])
            ax.set_xlim([left, right])
        fig.add_subplot(1, num_img, index, sharex=ax, sharey=ax)
        # opencv stores the color channels in the order of BGR so we need to reorder it to RGB
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')   # hide axis
    plt.show()

if __name__ == '__main__':
    visualization()
