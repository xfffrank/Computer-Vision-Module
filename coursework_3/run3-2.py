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
from helpers import FileHelper, ImageHelper, timeit, BagOfWords


class RunThree(BagOfWords):
    '''
    Our second method to explore the performance of run3.
    Sift features + bag of visual words + classifier(naive bayes or non-linear svm).
    '''

    def __init__(self, train_dir, test_dir, clusters=600, step_sz=5, batch_sz=2000, dense=True):
        '''The initialization of the class.
        
        :param int clusters: the number of clusters
        :param int batch_sz: the batch size for minibatch kmeans training
        :param int step_sz: the length of the gap between two sift descriptors
        :param boolean dense: If dense is True, dense sift descriptors will be used.
        '''
        BagOfWords.__init__(self, train_dir, test_dir)
        self.clusters = clusters
        self.step_sz = step_sz
        self.batch_sz = batch_sz
        self.dense = dense
        self.scaler = StandardScaler()
        self.image_h = ImageHelper()
        self.file_h = FileHelper()
        self.kmeans_model = None
        self.clf = None
        print('Current parameters:\n n_clusters: %s, batch_size: %s, step_size %s, dense: %s' 
            % (self.clusters, self.batch_sz, self.step_sz, self.dense))


    @timeit
    def preprocess_training_set(self):
        '''
        Find all the sift descriptors of the images in the training set.
        '''
        # get a dictionary containing images and corresponding labels
        img_dict = self.file_h.get_all_files(self.train_dir, 'train')
        all_des = []
        for label, img_list in img_dict.items():
            for img in img_list:
                des = self.image_h.gen_sift_des(img, self.dense, self.step_sz)
                all_des.append(des)
        # Stack arrays of sift descriptors in sequence vertically
        all_des = np.vstack(all_des)
        print('The shape of all sift descriptors:', all_des.shape)
        # fit all descriptors with a scaler
        self.scaler.fit(all_des)
        return self.scaler.transform(all_des)

    def generate_BOVW(self, image):
        '''
        generate bovw for one image
        '''
        # compute the sift features and normalize them
        dsift = self.image_h.gen_sift_des(image, self.dense, self.step_sz)
        # scale the features with the scaler that was trained with the training set
        dsift = self.scaler.transform(dsift)
        words = self.kmeans_model.predict(dsift)
        bovw = np.zeros(self.clusters, dtype=int)
        for w in words:
            bovw[w] += 1
        return bovw

    def measure_clf(self, clf, X, y, fold=10):
        '''
        Measure the performance of a classifier with 10-fold cross validation
        '''
        cv_score = cross_val_score(clf, X, y, cv=fold)
        print('Accuracy: %.2f' % (np.mean(cv_score)))

    @timeit
    def train(self):
        # get sift descriptors of all images
        des = self.preprocess_training_set()
        print('Start kmeans training...')
        self.kmeans_model = MiniBatchKMeans(n_clusters=self.clusters,
                                random_state=0, batch_size=self.batch_sz, compute_labels=False)
        self.kmeans_model.fit(des)
        print('Kmeans training completed.')
        df = self.BOVW_training_set()  # convert the training set to Bag of Visual Words
        labels = df.iloc[:, -1]  # get the labels of the images
        features = df.iloc[:, :-1]   # get the features of the images
        # clf = MultinomialNB()
        # self.measure_clf(clf, features, labels)
        # clf = GaussianNB()
        # self.measure_clf(clf, features, labels)
        # clf = BernoulliNB()
        # self.measure_clf(clf, features, labels)
        clf = SVC(kernel='poly')
        self.clf = clf.fit(features, labels)
        X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.33, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('score: ', score)

    @timeit
    def test(self):
        '''
        Predict the labels of the test set using the classifier that has been trained
        and ouput the run3.txt
        '''
        test_data, image_names = self.BOVW_testing_set()
        labels_predicted = self.clf.predict(test_data)
        with open('run3.txt', 'w') as f:
            for image, label in zip(image_names, labels_predicted):
                f.write(' '.join([image, label]) + '\n')

    def main(self):
        self.train()
        self.test()



if __name__ == '__main__':
    # tune parameters
    # for c in [150, 200, 250, 300]:
    #     for sz in [8, 9, 10, 11]:
    #         run = RunThree('../training/', '../testing/', clusters=c, step_sz=sz, batch_sz=2000, dense=True)
    #         run.main()
    
    # use the best parameters
    run = RunThree('../training/', '../testing/', clusters=200, step_sz=9, batch_sz=2000, dense=True)
    run.main()
        