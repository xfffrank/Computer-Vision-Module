import cv2
import os
import numpy as np
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
# helpers: self-defined module
from helpers import FileHelper, ImageHelper, timeit


class RunThree:
    '''
    Our first method to explore the performance of run3.
    Use dense sift and a classifier(naive bayes or non-linear svm).

    The performance is not as good as that of the model which we used in the first method
    so there is not a method which generates a result of the testing set here.
    '''
    def __init__(self, train_dir, resize_resolution, step_size=6):
        '''The initialization of the class.
        
        :param tuple resize_resolution: the resolution of the resized image
        :param int step_sz: the length of the gap between two sift descriptors
        '''
        self.train_dir = train_dir
        self.resize_r = resize_resolution
        self.image_h = ImageHelper()
        self.file_h = FileHelper()
        self.step_sz = step_size

    @timeit
    def preprocess_training_set(self):
        '''
        :return: the sift descriptors and labels of all the images in the training set
        '''
        # get a dictionary containing images and corresponding labels
        img_dict = self.file_h.get_all_files(self.train_dir, 'train')
        all_des = []
        all_labels = []
        for label, img_list in img_dict.items():
            for img in img_list:
                crop_img = self.image_h.crop_square(img)
                image = cv2.resize(crop_img, self.resize_r)
                des = self.image_h.gen_sift_des(image, dense=True, step_size=self.step_sz)
                # flatten the sift features into one vector for every image
                all_des.append(des.flatten())
                all_labels.append(label)
        return all_des, all_labels

    # @timeit
    # def train_naive_bayes(self, X, y):
    #     clf = MultinomialNB()
    #     cv = cross_val_score(clf, X, y, cv=10)
    #     print(cv)
    #     print(np.mean(cv))

    @timeit
    def train_svm(self, X, y):
        '''
        The training of svm takes much more time than naive bayes,
        so we used a part of the training set to evaluate the model
        instead of using cross validation.
        '''
        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.33, random_state=42)
        clf = SVC(kernel='poly')
        clf.fit(X_train, y_train)
        s = clf.score(X_test, y_test)
        print(s)

    def main(self):
        dataset, labels = self.preprocess_training_set()
        # use svm classifier
        self.train_svm(dataset, labels)
        



if __name__ == '__main__':
    resize_resolution = (180, 180)
    run = RunThree('../training/', resize_resolution, step_size=sz)
    run.main()









