import cv2
import os
import numpy as np
import math
import time
from abc import abstractmethod, ABC
from pandas import DataFrame


def timeit(func):
    '''
    A decorator which computes the time cost.
    '''
    def wrapper(*args, **kw):
        start = time.time()
        print('%s starts...' % (func.__name__))
        res = func(*args, **kw)
        print('%s completed: %.3f s' % (func.__name__, time.time() - start))
        return res
    return wrapper


class FileHelper:
    '''
    Operations related to files.
    '''
    def __init__(self):
        pass

    def get_all_files(self, directory, flag):
        '''
        Get all files from a directory.

        :param str directory: the directory that will be processed
        :param str flag: the directory type, 'train' or 'test'
        '''
        if flag == 'train':
            # return a dictionary which contains
            # all the images and their labels
            ret = dict()
            for f in os.listdir(directory):
                sub_dir = os.path.join(directory, f)
                if os.path.isdir(sub_dir):
                    img_path_list = [os.path.join(sub_dir, i) for i in os.listdir(sub_dir) if i.endswith('.jpg')]
                    # read every image with grayscale mode
                    img_list = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in img_path_list]
                    ret[f.lower()] = img_list
            return ret
        if flag == 'test':
            # return a tuple which contains all the images and their names
            # sort the list in numeric order using the filenames
            sorted_list = sorted(os.listdir(directory), key=lambda s: int(s.split('.')[0]))
            name_list = [i for i in sorted_list if i.endswith('.jpg')]
            img_path_list = [os.path.join(directory, i) for i in sorted_list if i.endswith('.jpg')]
            img_list = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in img_path_list]
            return img_list, name_list


class ImageHelper:
    '''
    Operations related to images.
    '''
    def __init__(self):
        # initialize a sift instance
        self.sift = cv2.xfeatures2d.SIFT_create()

    def crop_square(self, image):
        '''
        Crop an image to a square about the centre.
        '''
        rows, cols = image.shape
        if rows == cols:
            return image
        else:
            diff = abs(rows - cols)
            tmp = math.floor(diff / 2)
            if rows > cols:
                # The number of rows is bigger than the number of columns,
                # so the number of columns will be reserved.
                return image[tmp:tmp + cols, :]
            if rows < cols:
                # The number of columns is bigger than the number of rows,
                # so the number of rows will be reserved.
                return image[:, tmp:tmp + rows]

    def tiny_image(self, image, normalize):
        '''Create tiny image feature.
        
        Basically follow three steps:
            cropping, resizing, (normalizing)

        :param array image: The array representation of an image.
        :param boolean normalize: If set to True, the image will be made to have zero mean and unit length.
        :return: The feature vector of the image.
        :rtype: array
        '''
        crop_img = self.crop_square(image)
        resized_img = cv2.resize(crop_img, (16, 16))
        vector = resized_img.flatten()  # flatten the array
        if normalize:
            zero_mean_vec = vector - np.mean(vector)
            vector = zero_mean_vec / np.linalg.norm(zero_mean_vec)
        return vector

    def extract_patches(self, image, patch_size, stride):
        '''Extract 2-D patches from an image.
        Cut out the edges that cannot make up a patch.

        :param array image: the array presentation of an image.
        :param list patch_size: the dimensions of the patch.
        :param int stride: the length of the gap between the start of one patch and the start of the next consecutive patch
        :return: all the patches in the image
        :rtype: array
        '''
        rows, cols = image.shape
        p_row, p_col = patch_size
        patch_list = []
        for i in range(0, rows, stride):
            if i + p_row > rows:
                break  # cut out the edges of rows
            for j in range(0, cols, stride):
                if j + p_col > cols:
                    break  # cut out the edges of columns
                patch = image[i:i+p_row, j:j+p_col]
                patch_list.append(patch.flatten())   # flatten every 2-D patch into a vector
        return np.array(patch_list)

    def gen_sift_des(self, image, dense, step_size=5):
        '''
        Generate sift descriptors of the given image.

        :param boolean dense: If dense is True, dense sift descriptors will be used.
        :param int step_size: the length of the gap between two sift descriptors
        :return: all the sift descriptors of an image
        '''
        if dense:
            kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, image.shape[0], step_size) 
                                                for x in range(0, image.shape[1], step_size)]
            descriptors = self.sift.compute(image, kp)[1]
        else:
            descriptors = self.sift.detectAndCompute(image, None)[1]
        return descriptors

            
class BagOfWords(ABC):
    '''
    Operations related to Bag of Visual Words.
    '''
    def __init__(self, train_dir, test_dir):
        '''
        :param str train_dir: the directory of the training set
        :param str test_dir: the directory of the testing set
        '''
        self.file_h = FileHelper()
        self.train_dir = train_dir
        self.test_dir = test_dir

    @timeit
    def BOVW_training_set(self):
        '''Convert the training set to the presentation of Bag of Visual Words.
        
        Use a dataframe to save the features and labels of images
        which will be used in the  classifier.

        :return: a dataframe containing the features and image labels
        :rtype: DataFrame
        '''
        dataset = []
        # get the images and their labels from the training set
        img_dict = self.file_h.get_all_files(self.train_dir, 'train')
        for label, img_list in img_dict.items():
            for img in img_list:
                # generate the presentation of Bag of Visual Words for an image
                # using the trained kmeans model
                bovw = self.generate_BOVW(img)
                one_row = np.append(bovw, label)
                dataset.append(one_row)
        return DataFrame(dataset)   

    def BOVW_testing_set(self):
        '''Convert the testing set to the presentation of Bag of Visual Words.
        
        :return: A tuple containing the features of the testing set
                and corresponding image names.
        :rtype: tuple
        '''
        feature_set = []
        # get the images and their filenames from the testing set
        img_list, name_list = self.file_h.get_all_files(self.test_dir, 'test')
        for img in img_list:
            # generate the presentation of Bag of Visual Words for an image
            # using the trained kmeans model
            bovw = self.generate_BOVW(img)
            feature_set.append(bovw)
        return feature_set, name_list

    @abstractmethod
    def generate_BOVW(self, image):
        '''
        Generate the presentation of Bag of Visual Words for an image.

        Set it as an abstract method so it must be implemented in child classes.
        '''
        pass

              
        
