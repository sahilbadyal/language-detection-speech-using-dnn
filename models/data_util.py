#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import logging
import PIL.Image as Image
import numpy as np
from defs import LMAP
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename=__name__+'.log',format='%(levelname)s:%(message)s', level=logging.DEBUG)


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

    def vectorize(self, examples,augm=False):
            #print(len(examples))
            im_data = np.zeros((len(examples), self.n_channels, self.y_features, self.x_features), dtype=np.float32)
            labels = []
            for i,example in enumerate(examples):
                    #print example
                    im = self.read_png(example[0])
                    im_array = np.array(im).astype(np.float32)
                    if augm:
                            offset = random.randint(0, 90)
                            im_data[i, 0, :, :] = im_array[:self.y_features, offset:offset+self.x_features] / 256.0
                    else:
                            im_data[i, 0, :, :] = im_array[:self.y_features, :] / 256.0
                    labels.append(LMAP[int(example[1])])
            return im_data,labels

    @classmethod
    def build(cls, data):
            return cls(data.n_channels, data.x_features, data.y_features,data.png_folder,data.n_classes)


    def load_and_preprocess_data(self,examples):
            #print(examples[0])
            #logger.info("Loading  data...%d ",len(examples))
            
            # now process all the input data.
            augm = False
            inputs,labels = self.vectorize(examples,augm)
            
            #logger.info("Done reading %d images", len(examples))

            return inputs,labels

    def read_png(self,image):
            return Image.open(self.base_path+str(image)+'.png')

def getModelHelper(args):
        helper = ModelHelper.build(args)
        return helper


class Config:
        """Holdsmodelhyperparamsanddatainformation.

        Theconfigclassisusedtostorevarioushyperparametersanddataset
        informationparameters.ModelobjectsarepassedaConfig()objectat
        instantiation.
        """
        n_channels=1
        x_features=678
        y_features=128
        n_classes=3
        dropout=0.5
        batch_size=32
        n_epochs=1
        lr=0.001
        batch_size=32
        lstm_size=500
        png_folder='../data/train/pngaugm/'

def testModelHelper():

        config = Config()

        helper = getModelHelper(config)
        
        list_ = []

        with open('../data/valEqualAugm.csv') as valFile:
                lines = valFile.readlines()
                for line in lines:
                        split = line.split(',')
                        list_.append((split[0],split[1]))

        #list_ = [(l,m) for l,m in zip(list1,list2) ] 

        in_data,labels  = helper.load_and_preprocess_data(list_[2801:3000])

        y = {k:i for k,i in zip(list_,labels)}
        print in_data[100],labels[100],list_[2900]

        assert np.shape(in_data)== (8,1,128,858),np.shape(labels)==(8,3,)
if __name__ == "__main__":
        testModelHelper()

