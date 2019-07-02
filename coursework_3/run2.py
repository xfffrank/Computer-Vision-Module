import cv2
import numpy as np
import time
import os
import sklearn
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
import pandas as pd
# helpers: self-defined module
from helpers import ImageHelper, FileHelper, timeit, BagOfWords


class RunTwo(BagOfWords):

    def __init__(self, train_dir, test_dir, n_c=600, batch_sz=5000, patch_size=[6, 6], stride=4):
        '''The initialization of the class.
        
        :param int n_c: the number of clusters
        :param int batch_sz: the batch size for minibatch kmeans training
        :param list patch_size: the dimensions of the patch
        :param int stride: the length of the gap between the start of one patch and the start of the next consecutive patch
        '''
        BagOfWords.__init__(self, train_dir, test_dir)
        self.n_clusters = n_c
        self.batch_size = batch_sz 
        self.patch_size = patch_size
        self.stride = stride
        self.image_h = ImageHelper()
        self.file_h = FileHelper()
        self.kmeans_model = None
        self.clf = None
        print('Current parameters:\n n_clusters: %s, batch_size: %s, patch_size %s, stride: %s' 
            % (self.n_clusters, self.batch_size, self.patch_size, self.stride))

    @timeit
    def preprocess_training_set(self):
        '''Find all the patches of the images in the training set.

        Extract patches and normalize them.
        '''
        # get the images and their labels from the training set
        img_dict = self.file_h.get_all_files(self.train_dir, 'train')
        all_patches = []
        for label, img_list in img_dict.items():
            for img in img_list:
                patches = self.image_h.extract_patches(img, self.patch_size, self.stride)
                all_patches.append(patches)
        # Stack arrays of patches in sequence vertically
        all_patches = np.vstack(all_patches)
        print('The shape of all patches: ', all_patches.shape)
        # mean-centring and nomalizing each patch
        normalized = preprocessing.scale(np.float64(all_patches), axis=1)
        return normalized

    def generate_BOVW(self, image):
        '''Convert an image to the presentation of Bag of Visual Words.

        :param numpy.ndarray image: the given image
        '''
        patches = self.image_h.extract_patches(image, self.patch_size, self.stride)
        normalized = preprocessing.scale(np.float64(patches), axis=1)
        words = self.kmeans_model.predict(normalized)
        bovw = np.zeros(self.n_clusters, dtype=int)
        for w in words:
            bovw[w] += 1
        return bovw


    @timeit
    def train(self):
        '''The training process for run2.

        1. Use MiniBatchKmeans to learn a vocabulary.
        2. Convert the training set using the presentation of Bag of Visual Words.
        3. Build a linear classifier and measure its performance.
        '''
        print('Training starts...')
        training_patches = self.preprocess_training_set()
        print('Start kmeans training...')
        # save the trained kmeans model for the use of testing set
        self.kmeans_model = MiniBatchKMeans(n_clusters=self.n_clusters,
                                random_state=0, batch_size=self.batch_size, compute_labels=False)
        self.kmeans_model.fit(training_patches)
        del training_patches  # release the memory used
        print('Kmeans training completed.')
        train_data = self.BOVW_training_set()
        print('Start building a classifier...')
        labels = train_data.iloc[:, -1]  # get the labels of the images
        features = train_data.iloc[:, :-1]  # get the features of the images
        clf = LogisticRegression(random_state=0, multi_class='ovr', solver='sag', n_jobs=-1)
        # measure the performance of the model with cross validation
        cv_score = cross_val_score(clf, features, labels, cv=10)
        print('Average score of 10-fold cross validation for LR: %.2f' % np.mean(cv_score))
        # feed the model with all the training set
        clf.fit(features, labels)
        # # save the trained model for predicting testing set
        self.clf = clf

    @timeit
    def test(self):
        '''
        Predict the labels of the test set using the classifier that has been trained
        and ouput the run2.txt
        '''
        test_data, image_names = self.BOVW_testing_set()
        labels_predicted = self.clf.predict(test_data)
        with open('run2.txt', 'w') as f:
            for image, label in zip(image_names, labels_predicted):
                f.write(' '.join([image, label]) + '\n')

    def main(self):
        self.train()
        self.test()


if __name__ == '__main__':
    run = RunTwo(train_dir='../training/', test_dir='../testing/', n_c=700, batch_sz=2000, patch_size=[6,6], stride=3)
    run.main()
