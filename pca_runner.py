# coding:utf-8
import gzip
import six
import time
import pcanet
import numpy as np
from sklearn import *


def load_data(data_set):
    import os
    data_dir, data_file = os.path.split(data_set)
    if data_dir == "" and not os.path.isfile(data_set):
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", data_set)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            data_set = new_path

    if (not os.path.isfile(data_set)) and data_file == 'mnist.pkl.gz':
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % origin)
        six.moves.urllib.request.urlretrieve(origin, data_set)

    print('... loading data ...')

    with gzip.open(data_set, 'rb') as f:
        try:
            train_set, valid_set, test_set = six.moves.cPickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = six.moves.cPickle.load(f)
    return train_set, valid_set, test_set


class PCANetPara(object):

    def __init__(self):
        self.patch_size = [5, 5, 5]
        self.num_filters = [8, 8, 5]
        self.num_stages = len(self.patch_size)

        self.hist_block_size = [5, 5]
        self.blk_overlap_ratio = 0.5

    pass


class Runner:

    def __init__(self):

        # read data
        test_data, _, train_data = load_data('mnist.pkl.gz')
        self.train_x, self.train_y = train_data
        self.test_x, self.test_y = test_data

        # reshape
        self.train_x = self.train_x.reshape(self.train_x.shape[0], 28, 28, 1)
        self.test_x = self.test_x.reshape(self.test_x.shape[0], 28, 28, 1)

        # pca para
        self.pac_net_para = PCANetPara()

        # about train
        self.train_size = len(self.train_x)
        self.train_result = None
        self.v = None

        # about classifies
        self.svm_model = None

        # about test
        self.accuracy = 0
        self.accuracy_number = 0

        # time
        self.pca_net_time = 0
        self.svm_classifier_time = 0
        self.test_time = 0

        pass

    def run(self, train_number=100, test_number=100, train_display_number=20, test_display_number=5):
        self.train(train_size=train_number, display_number=train_display_number)
        self.classifier()
        self.predict(test_number=test_number, display_number=test_display_number)
        self.print_info()

    def run_all(self):
        self.train(train_size=len(self.train_x))
        self.classifier()
        self.predict(test_number=len(self.test_x), display_number=len(self.test_x)//1000)
        self.print_info()

    def train(self, train_size, display_number):
        start = time.time()
        self.train_size = np.min([train_size, self.train_size])
        self.train_result, self.v = pcanet.pca_net_train(self.train_x[: self.train_size], self.pac_net_para, 1, display_number)
        self.pca_net_time = time.time() - start
        print('pca_net training time:%f' % self.pca_net_time)

        pass

    def classifier(self):
        start = time.time()
        self.svm_model = svm.LinearSVC().fit(self.train_result, self.train_y[: self.train_size])
        self.svm_classifier_time = time.time() - start
        print('SVM classifier training time:%f' % self.svm_classifier_time)

        pass

    def predict(self, test_number, display_number):
        test_number = np.min([len(self.test_x), test_number])
        # 由于有问题，所以先这样
        batch_size = 1
        batch_number = test_number // batch_size
        all_test_number = batch_size * batch_number
        print("all batch number is %d/%d" % (batch_number, all_test_number))

        start = time.time()
        for i in range(batch_number):
            # data
            test_x_i_batch = self.test_x[i * batch_size: (i + 1) * batch_size].reshape((batch_size,) + self.test_x[i].shape)

            # feature ext and predict
            test_feature = pcanet.pca_net_fea_ext(self.pac_net_para, test_x_i_batch, self.v)
            label_predict = self.svm_model.predict(test_feature)

            # stat accuracy
            for index, label in enumerate(label_predict):
                if label == self.test_y[i * batch_size + index]:
                    self.accuracy_number += 1
                pass

            # progress
            if i % display_number == 0:
                print(time.strftime("%H:%M:%S", time.localtime()), "{}/{} {}".format(i, batch_number, self.accuracy_number))
            pass

        # calculate accuracy
        self.accuracy = self.accuracy_number / all_test_number

        # print
        self.test_time = time.time() - start
        print('testing time: {}'.format(self.test_time))
        print('test accuracy {} ({}/{})'.format(self.accuracy, self.accuracy_number, all_test_number))

    def print_info(self):
        print('All time:  {} secs.'.format(self.pca_net_time + self.svm_classifier_time + self.test_time))
        print('Testing accuracy:  {}%'.format(100 * self.accuracy))


if __name__ == '__main__':

    Runner().run()
    # Runner().run_all()

    pass
