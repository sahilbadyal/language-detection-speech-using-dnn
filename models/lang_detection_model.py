#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
A model for language detection in speech
"""
import logging

import tensorflow as tf
from models.base_model import Model
from models.util import ConfusionMatrix, Progbar, minibatches
#from models.data_util import get_chunks
from models.defs import LBLS
import numpy as np

logger = logging.getLogger("langDec.model")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class LangDetectionModel(Model):
    """
    Implements special functionality for Spoken Language Detection models.
    """

    def __init__(self, helper, config):
        self.helper = helper
        self.config = config

    def preprocess_speech_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of spectrograms/MFCC and corresponding language input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def evaluate(self, sess, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting input as language entities.
        """
        class_cm = ConfusionMatrix(labels=LBLS)
        
        total_images = set()
        correct_preds = set()
        correct_preds, total_correct, total_preds = 0., 0., 0.

        #need to correct
        for _, actual, pred  in self.output(sess, examples_raw):
            #print(actual,pred)
            class_cm.update(actual, pred)
            #total_images.
            #actual = set(get_chunks(labels))
            #pred = set(get_chunks(labels_))
  
            #correct_preds += len(actual.intersection(pred))
            #total_preds += len(pred)
            #total_actual += len(actual)

        #p = correct_preds / total_preds if correct_preds > 0 else 0
        #r = correct_preds / total_correct if correct_preds > 0 else 0

        #f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return class_cm


    def output(self, sess, inputs_raw):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        inputs = []
        preds = []
        val_loss = []
        prog = Progbar(target=1 + int(len(inputs_raw) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs_raw, self.config.batch_size, shuffle=False)):
            inputs_,labels = self.preprocess_speech_data(batch)
            preds_,_loss,_summary = self.predict_on_batch(sess, inputs_,labels)
            preds += list(preds_)
            inputs += list(inputs_)
            prog.update(i + 1, [])
            self.val_writer.add_summary(_summary,i)
            val_loss.append(_loss)
        logger.info("Mean Val loss  = %.2f ",np.mean(val_loss))
        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        best_score = 0.
        step = 0

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            # You may use the progress bar to monitor the training progress
            # Addition of progress bar will not be graded, but may help when debugging
            n_minibatches = 1 + int(len(train_examples_raw) / self.config.batch_size)
            prog = Progbar(target=n_minibatches)
			# The general idea is to loop over minibatches from train_examples, and run train_on_batch inside the loop
			# Hint: train_examples could be a list containing the feature data and label data
			# Read the doc for utils.get_minibatches to find out how to use it.
                        # Note that get_minibatches could either return a list, or a list of list
                        # [features, labels]. This makes expanding tuples into arguments (* operator) handy
            epoch_loss = []
            for i,batch in enumerate(minibatches(train_examples_raw,self.config.batch_size)):
                    inputs,labels = self.preprocess_speech_data(batch)
                    loss,_summary = self.train_on_batch(sess,inputs,labels)
                    prog.update(i+1,[("loss",loss)])
                    self.train_writer.add_summary(_summary,step)
                    epoch_loss.append(loss)
                    step += 1

            logger.info("Epoc loss after epoch %d = %.2f",epoch,np.mean(epoch_loss))

            logger.info("Evaluating on development data")
            language_cm = self.evaluate(sess, dev_set_raw)
            logger.debug("Lang-level confusion matrix:\n" + language_cm.as_table())
            logger.debug("Lang-level scores:\n" + language_cm.summary())
            #logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

            #score = entity_scores[-1]
            
            #if score > best_score:
            #    best_score = score
            #    if saver:
            #        logger.info("New best score! Saving model in %s", self.config.model_output)
            #        saver.save(sess, self.config.model_output)
            print("")
        return best_score
