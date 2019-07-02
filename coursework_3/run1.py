import cv2
import os
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
# helpers: self-defined module
from helpers import ImageHelper, FileHelper


class ImageClassifierKNN:

    def __init__(self, train_dir, test_dir, normalize, K_range):
        '''Initialization of the class.

        :param str train_dir: the directory of the training set;
        :param str test_dir: the directory of the test set;
        :param boolean normalize: If set to True, the image will be made to have zero mean and unit length.
        :param int K_range: the range of K values that will be used in KNN training.
        '''
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.normalize = normalize
        self.K_range = K_range
        self.image_h = ImageHelper()
        self.file_h = FileHelper()

    def preprocess_train_dataset(self):
        '''
        Preprocess the training set.

        :return: a dataframe which contains the features and the corresponding label.
        :rtype: DataFrame
        '''
        # get the images and their labels from the training set
        img_dict = self.file_h.get_all_files(self.train_dir, 'train')
        dataset = []
        for label, img_list in img_dict.items():
            for img in img_list:
                # get the tiny image feature vector
                vector = self.image_h.tiny_image(img, self.normalize)
                row = np.append(vector, label)
                dataset.append(row)
        df_training = DataFrame(dataset)
        return df_training

    def preprocess_test_dataset(self):
        '''
        Preprocess the test set.

        :return: A tuple containing the features of the testing set
                and corresponding image names.
        :rtype: tuple
        '''
        feature_set = []
        # get the images and their filenames from the testing set
        img_list, name_list = self.file_h.get_all_files(self.test_dir, 'test')
        for img in img_list:
            vector = self.image_h.tiny_image(img, self.normalize)
            feature_set.append(vector)
        return feature_set, name_list

    def train(self, k_range):
        '''
        Use GridSearchCV to train models and find the best model.
        '''
        print('preprocessing training dataset...')
        df = self.preprocess_train_dataset()
        # get the labels of the images
        label = df.iloc[:, -1]
        # get the features of the images
        dataset = df.iloc[:, :-1]
        # specify the range of K values for KNN
        K_values = list(range(1, k_range))
        # prepare the parameters for GridSearchCV
        params = dict(n_neighbors=K_values)
        knn = KNeighborsClassifier()
        clf = GridSearchCV(knn, params, cv=10, scoring='accuracy', refit=True)
        print('The training process begins...')
        clf.fit(dataset, label)
        print('The best model found!')
        best_k = clf.best_params_['n_neighbors']
        best_score = clf.best_score_
        # save the best model for later use in testing set
        self.best_model = clf.best_estimator_
        print('optimal k value:', best_k)
        print('best score:', best_score)

    def test(self):
        '''
        Predict the labels of the test set using the best model based on GridSearchCV
        and ouput the run1.txt
        '''
        test_features, images = self.preprocess_test_dataset()
        labels_predicted = self.best_model.predict(test_features)
        with open('run1.txt', 'w') as f:
            for image, label in zip(images, labels_predicted):
                f.write(' '.join([image, label]) + '\n')

    def main(self):
        ''' 
        Run the routines for the run1 task
        '''
        # Use the training dataset to train and find the optimal K value.
        self.train(self.K_range)
        # Predict the labels of the test set with the best model
        # and output the result.
        self.test()


if __name__ == '__main__':
    classifier = ImageClassifierKNN('../training/', '../testing/', True, K_range=10)
    classifier.main()


