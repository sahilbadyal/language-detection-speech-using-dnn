#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import logging
import PIL.Image as Image
import numpy as np
from defs import LBLS
from util import one_hot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def evaluate(model, X, Y):
    cm = ConfusionMatrix(labels=LBLS)
    Y_ = model.predict(X)
    for i in range(Y.shape[0]):
        y, y_ = np.argmax(Y[i]), np.argmax(Y_[i])
        cm.update(y,y_)
    cm.print_table()
    return cm.summary()

class ModelHelper(object):
    """
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    def __init__(self,n_channels,x_features,y_features,base_path,n_classes):
        self.n_channels = n_channels
        self.x_features = x_features
        self.y_features = y_features
        self.base_path = base_path
        self.n_classes = n_classes

    def vectorize(self, examples):
            im_data = np.zeros((len(examples), self.n_channels, self.x_features, self.y_features), dtype=np.float32)
            labels = []
            for i,example in enumerate(examples):
                    im = self.read_png(example[0])
                    im_array = np.array(im).astype(np.float32)
                    im_data[i, 0, :, :] = im_array[:self.x_features, :] / 256.0
                    labels.append(one_hot(self.n_classes,int(example[1])))
            return im_data,labels

    @classmethod
    def build(cls, data):
            return cls(data['n_channels'], data['x_features'], data['y_features'],data['base_path'],data['n_classes'])


    def load_and_preprocess_data(self,examples):
            print(examples[0])
            logger.info("Loading  data...%d ",len(examples))
            
            # now process all the input data.
            inputs,labels = self.vectorize(examples)
            
            logger.info("Done reading %d images", len(examples))

            return inputs,labels

    def read_png(self,image):
            return Image.open(self.base_path+str(image)+'.png')

def getModelHelper(args):
        helper = ModelHelper.build(args)
        return helper


def testModelHelper():
        args = {
                'n_channels':1,
                'x_features':128,
                'y_features':858,
                'base_path':'../data/train/png/',
                'n_classes':3
        }

        helper = getModelHelper(args)

        list1 = ['0aaatsgitup', '0aardcafriy', '0aaus1iol2h', '0aboxbqzx2n', '0abyp4czxgr', '0acbkn5sl3x', '0acwgcalkob', '0acwmldxgzf']

        list2 = ['0','1','2','1','0','1','2','0']

        list_ = [(l,m) for l,m in zip(list1,list2) ] 

        in_data,labels  = helper.load_and_preprocess_data(list_)

        assert np.shape(in_data)== (8,1,128,858),np.shape(labels)==(8,3,)
if __name__ == "__main__":
        testModelHelper()

