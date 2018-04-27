#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A combination of COVNET AND RNN nets for Lang Detection
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from util import print_sentence
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from lang_detection_model import LangDetectionModel
from defs import LBLS

logger = logging.getLogger("ld.cov.rnn")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_channels = 1
    x_features = 128
    y_features = 858 # Number of features for every word in the input.
    max_length = 120 # longest sequence to parse
    n_classes = 3
    dropout = 0.5
    embed_size = 50
    hidden_size = 300
    batch_size = 32
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001

class COVRNNModel(LangDetectionModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        self.input_placeholder = tf.placeholder(shape=[None,self.config.n_channels,self.config.x_features,self.config.y_features],dtype=tf.float32)
        self.labels_placeholder = tf.placeholder(shape=[None,self.config.n_classes],dtype=tf.int32)
        self.dropout_placeholder = tf.placeholder(shape=(),dtype=tf.float32)

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=0):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {
                self.input_placeholder:inputs_batch
                }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] =  dropout
        return feed_dict
    
    def add_prediction_op(self):
        """Adds the following network:

        Returns:
            pred: tf.Tensor of shape (batch_size,)
        """
        input_layer  = self.input_placeholder

        dropout_rate = self.dropout_placeholder

        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        xavier_init = tf.contrib.layers.xavier_initializer()
        init = tf.constant_initializer(0)
        U = tf.get_variable(name='U',shape=[self.config.hidden_size,self.config.n_classes],dtype=tf.float32,initializer=xavier_init)
        b2 = tf.get_variable(name='b2',shape=[self.config.n_classes],dtype=tf.float32,initializer=init) 
        h_0 =  tf.zeros(shape=[tf.shape(input_layer)[0],self.config.hidden_size],dtype=tf.float32)

        conv1 = tf.layers.conv2d(
                      inputs=input_layer,
                      filters=32,
                      kernel_size=[3, 3],
                      padding="same",
                      activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
        
        #Add batch Norm
        output_cnn = tf.contrib.layers.batch_norm(pool1, center=True, scale=True, is_training=self.isTraining,scope='bn_2')


        conv2 = tf.layers.conv2d(
                      inputs=pool1,
                      filters=32,
                      kernel_size=[3, 3],
                      padding="same",
                      activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

        #add batch norm

        output_cnn = tf.contrib.layers.batch_norm(pool2, center=True, scale=True, is_training=self.isTraining,scope='bn_2')

        num_channels  = 32
        filter_W = 54
        filter_H = 8

        # We squezed all data channel wise and fed it into an rnn
        channels = []
        for channel_index in range(num_channels):
                channels.append(output[:, channel_index, :, :].transpose((0, 2, 1)))

        ##Add rnn here
        with tf.name_scope('lstm_layer'):
                lstm = tf.contrib.rnn.BasicLSTMCell(self.config.lstm_size)
        with tf.name_scope('dropout'):
                cell = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1-dropout_rate)
        

        for i,channel in enumerate(channels):
                if(i==0):
                        initial_state = cell.zero_state(tf.shape(channel)[0] , tf.float32)
                        outputs, state = cell(inputs=channel,state=initial_state)
                 else:
                        outputs, state = cell(inputs=channel,state=state)
        
        output_rnn = tf.stack(outputs,axis=1)

        ##final batch normalization
        
        outputs = tf.contrib.layers.batch_norm(output_rnn, center=True, scale=True, is_training=self.isTraining,scope='bn_3')


        ##Add fully connected layer
         with tf.name_scope('affine_layer'):
                 preds = tf.contrib.layers.fully_connected(outputs,self.config.n_class, activation_fn=None)

        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels_placeholder,logits=preds))
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(Config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def preprocess_speech_data(self, examples):
        def featurize_windows(data, start, end, window_size = 1):
            """Uses the input sequences in @data to construct new windowed data points.
            """
            ret = []
            for sentence, labels in data:
                from util import window_iterator
                sentence_ = []
                for window in window_iterator(sentence, window_size, beg=start, end=end):
                    sentence_.append(sum(window, []))
                ret.append((sentence_, labels))
            return ret

        examples = featurize_windows(examples, self.helper.START, self.helper.END)
        return pad_sequences(examples, self.max_length)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            assert len(labels_) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=Config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(RNNModel, self).__init__(helper, config, report)
        self.max_length = min(Config.max_length, helper.max_length)
        Config.max_length = self.max_length # Just in case people make a mistake.
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        self.build()

def do_test2(args):
    logger.info("Testing implementation of RNNModel")
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)

    logger.info("Model did not crash!")
    logger.info("Passed!")

def do_train(args):
    # Set up some parameters.
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None #Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev_raw))
                report.save()
            else:
                # Save predictions in a text file.
                output = model.output(session, dev_raw)
                sentences, labels, predictions = zip(*output)
                predictions = [[LBLS[l] for l in preds] for preds in predictions]
                output = zip(sentences, labels, predictions)

                with open(model.config.conll_output, 'w') as f:
                    write_conll(f, output)
                with open(model.config.eval_output, 'w') as f:
                    for sentence, labels, predictions in output:
                        print_sentence(f, sentence, labels, predictions)

def do_evaluate(args):
    config = Config(args)
    helper = ModelHelper.load(args.model_path)
    input_data = read_conll(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            for sentence, labels, predictions in model.output(session, input_data):
                predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions)
