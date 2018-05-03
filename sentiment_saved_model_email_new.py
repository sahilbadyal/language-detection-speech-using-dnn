#!/usr/bin/python3

import os
print('Curr Dir:',os.getcwd())
import sys
import tensorflow as tf

import numpy as np
import tensorflow as tf
import re
from collections import Counter
import json
from pprint import pprint
from tensorflow.contrib import learn
import re
import csv
import pickle
from tensorflow.python.framework import ops
from model_utilities import *
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS
TRAIN_FILE_NAME = 'train_set/emailSentimentTrainDataset.v.01.polar.tsv'
VALID_FILE_NAME = 'val_set/emailSentimentValDataset.v.01.polar.tsv'
TEST_FILE_NAME = 'test_set/emailSentimentTestDataset.v.01.tsv'
GLOVE_FILE_NAME = 'glove.840B.300d.txt'
NUMBER_OF_GLOVE_FEATURES = 300
NUMBER_OF_LABELS = 1
MAX_SEQ_LENGTH = 100
KEEP_PROB = 0.5

lstm_size_list = [64] #[16,32,64]
lstm_layers_list = [1]
batch_size_list = [256] #[16,32,64,128]
n_epochs_list = [6] #[10,15,20]
LEARNING_RATE = 0.001

hyperpara_index = 0

MODEL_SIGNATURE_NAME = 'predict_sentiment_email'

def main(_):
    x_ = []
    loss_count = 0
    train_loss = []
    val_loss = []
    
    if FLAGS.training_iteration <= 0:
        print 'Please specify a positive value for training iteration.'
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print 'Please specify a positive value for version number.'
        sys.exit(-1)

    print 'Training model...'

    sentence_train,score_train,cat_train = load_input_and_labels(TRAIN_FILE_NAME)
    sentence_valid,score_valid,cat_val = load_input_and_labels(VALID_FILE_NAME)
    sentence_test,score_test,cat_test = load_input_and_labels(TEST_FILE_NAME)        
    
    print len(sentence_train)
    print len(score_train)
    print len(sentence_valid)
    print len(score_valid)
    print len(sentence_test)
    print len(score_test)
    print "Train, valid and test files loaded" 
    
    vocab,embd = loadGloVe(GLOVE_FILE_NAME)
    #print vocab[:2]
    #print embd[:2]
    print len(vocab)
    print len(embd)
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    #vocab = set(vocab)
    word_to_int = {word:i for i,word in enumerate(vocab)}
    vocab = set(vocab)
    #word_to_int["<None>"] = 0
    print word_to_int["<None>"]
    print word_to_int["<email>"]
    print word_to_int["<phone>"]
    print "Glove file loaded"


    features_train = convert_to_features(sentence_train,vocab,word_to_int)
    features_test = convert_to_features(sentence_test,vocab,word_to_int)
    features_val = convert_to_features(sentence_valid,vocab,word_to_int)
    #features_dcs = convert_to_features(sentence_dcs,vocab,word_to_int)
    #features_studio = convert_to_features(sentence_studio,vocab,word_to_int)

    train_x,test_x,val_x = get_lookup_features(vocab_size, embedding, embedding_dim,features_train, features_test, features_val)
    #dcs_x,studio_x,val_x_ = get_lookup_features(vocab_size, embedding, embedding_dim,features_dcs, features_studio, features_val)
    
        
    print "Features formed"
    
    train_y = score_train
    test_y = score_test
    val_y = score_valid
    test_data = list(zip(test_x,test_y,cat_test))
    dcs_x,dcs_y = getCatWiseData('DCS',test_data)
    studio_x,studio_y = getCatWiseData('STUDIO',test_data)
    #print train_x[:2]

    all_hyperpara_list = get_hyperparamters_list(lstm_size_list,lstm_layers_list, batch_size_list, n_epochs_list)

    print "Hyperparamter Tuning started"

    hyperpara_index = 0 #Hyperparameter S.No.

    for hyperameter_tuple in all_hyperpara_list:  
        print(hyperameter_tuple)
        lstm_size = hyperameter_tuple[0]
        lstm_layers = hyperameter_tuple[1]
        batch_size = hyperameter_tuple[2]
        n_epochs = hyperameter_tuple[3]
        fileName = 'hyper_'+str(hyperpara_index)+'_'+str(lstm_size)+'_'+str(batch_size)+'_'+str(n_epochs)
        fileNameLoss = fileName + '_loss'
        fileName = fileName + '.csv'
        fileNameLoss = fileNameLoss + '.csv'
        file_writer(fileName,['lstm_size= '+str(lstm_size), 'lstm_layers= '+str(lstm_layers),'batch_size= '+str(batch_size),'n_epochs= '+str(n_epochs)])
        ops.reset_default_graph()
        
        X,Y,keep_prob = get_input_and_label_tensor()
        
        lstm,cell,initial_state, outputs, final_state, predictions = build_network(lstm_size,lstm_layers,X,keep_prob)
    
        loss, optimizer = optimize_network(Y,predictions)
    
        correct_pred, accuracy = get_accuracy(Y,predictions)
    
        #START SESSION:
        sess = tf.Session()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train_1',
                                      sess.graph)
        test_writer = tf.summary.FileWriter('./test_1')
        val_writer = tf.summary.FileWriter('./val_1')
        #dcs_writer = tf.summary.FileWriter('./dcs_1')
        #studio_writer = tf.summary.FileWriter('./studio_1')
        sess.run(tf.global_variables_initializer())
        
        with tf.device('/cpu:0'):
            steps = 0
            #For every Epoch
            numberOfBatches = 0
            for e in range(n_epochs):
                count_=0
                batch_index = 1
                
                temp_train_loss = []
                #For every batch
                for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                    if(count_==0):
                        state = sess.run(initial_state, feed_dict={X:x})
                    #print(x)
                    feed = {X: x, Y: np.reshape(y,(len(y),1)), keep_prob: KEEP_PROB,initial_state: state}
                    state, loss_,  _ = sess.run([final_state, loss, optimizer], feed_dict=feed)
                    temp_train_loss.append(loss_)
                    if batch_index%50==0:
    		    #diff_pred = Y - predictions
    		    #square_diff_pred = tf.square(diff_pred)
    		    #output_, out_y, out_pred, square_pred, value_pred = sess.run([outputs,Y,predictions,square_diff_pred, diff_pred], feed_dict=feed)
                        #print("y is = ", out_y)
    		    #print("pred is = ", out_pred)
    		    #print("value_pred = ", value_pred)
    	            #print("square pred = ",square_pred)
    		    #print("output = ",output_)
                        temp_val_loss = []
                        val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                        for x, y in get_batches(val_x, val_y, batch_size): 
                            feed = {X: x,Y: np.reshape(y,(len(y),1)),keep_prob:1,initial_state: val_state}
                            loss_, val_state = sess.run([loss, final_state], feed_dict=feed)
                            temp_val_loss.append(loss_)
                        mean_val_loss = np.mean(temp_val_loss)
                        mean_train_loss = np.mean(temp_train_loss)
                        val_loss.append(mean_val_loss)  
                        print("Epoch: {}/{}".format(e, n_epochs),"Iteration: {}".format(batch_index),"Train loss: {:.3f}".format(mean_train_loss),"Val Loss: {:.3f}".format(mean_val_loss))
                          
                    if steps%10==0:
                        val_acc = []
                        val_loss = []
                        val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                        for x, y in get_batches(val_x, val_y, batch_size):
                            feed = {X: x,Y: np.reshape(y,(len(y),1)),keep_prob:1,initial_state: val_state}
                            summary,batch_loss,batch_acc, val_state = sess.run([merged,loss,accuracy, final_state], feed_dict=feed)
                            val_writer.add_summary(summary,steps)
                            val_acc.append(batch_acc)
                            val_loss.append(batch_loss)
                        print("Val acc: {:.5f}".format(np.mean(val_acc)))
                        #file_writer(fileName,['Epoch= '+str(e),'Validation Acc= '+str(np.mean(val_acc)) ])
                        #file_writer(fileNameLoss,['Epoch= '+str(e),'Validation Loss= '+str(np.mean(val_loss)) ])
                         #Train Acc:
                        train_acc = []
                        train_loss = []
                        train_state = sess.run(cell.zero_state(batch_size, tf.float32))
                        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                            feed = {X: x,Y:  np.reshape(y,(len(y),1)),keep_prob:1,initial_state: train_state}
                            summary,batch_loss,batch_acc, train_state = sess.run([merged,loss,accuracy, final_state], feed_dict=feed)
                            train_writer.add_summary(summary,steps)
                            train_acc.append(batch_acc)
                            train_loss.append(batch_loss)
                        print("Train accuracy: {:.5f}".format(np.mean(train_acc)))
                        #file_writer(fileName,['Epoch= '+str(e),'Train Acc= '+str(np.mean(train_acc)) ])
                        #file_writer(fileNameLoss,['Epoch= '+str(e),'Train Loss= '+str(np.mean(train_loss)) ])
                        #Test Acc:
                        test_acc = []
                        test_loss = []
                        test_state = sess.run(cell.zero_state(batch_size, tf.float32))
                        for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
                            feed = {X: x,Y: np.reshape(y,(len(y),1)),keep_prob:1,initial_state: test_state}
                            summary,batch_loss,batch_acc, test_state = sess.run([merged,loss,accuracy, final_state], feed_dict=feed)
                            train_writer.add_summary(summary,steps)
                            test_acc.append(batch_acc)
                            test_loss.append(batch_loss)
                        print("Test accuracy: {:.5f}".format(np.mean(test_acc)))
                        file_writer(fileName,['Epoch= '+str(e),'Test Acc= '+str(np.mean(test_acc)) ])
                        file_writer(fileNameLoss,['Epoch= '+str(e),'Test Loss= '+str(np.mean(test_loss)) ])
                        #DCS Acc:                        
                        dcs_acc = []
                        dcs_loss = []
                        dcs_state = sess.run(cell.zero_state(batch_size, tf.float32))
                        for ii, (x, y) in enumerate(get_batches(dcs_x, dcs_y, batch_size), 1):
                            feed = {X: x,Y: np.reshape(y,(len(y),1)),keep_prob:1,initial_state: dcs_state}
                            batch_loss,batch_acc, dcs_state = sess.run([loss,accuracy, final_state], feed_dict=feed)
                            dcs_acc.append(batch_acc)
                            dcs_loss.append(batch_loss)
                        print("DCS accuracy: {:.5f}".format(np.mean(dcs_acc)))
                        file_writer(fileName,['Epoch= '+str(e),'DCS Acc= '+str(np.mean(dcs_acc)) ])
                        file_writer(fileNameLoss,['Epoch= '+str(e),'DCS Loss= '+str(np.mean(dcs_loss)) ])
                        #Studio Acc:
                        studio_acc = []
                        studio_loss = []
                        studio_state = sess.run(cell.zero_state(batch_size, tf.float32))
                        for ii, (x, y) in enumerate(get_batches(studio_x, studio_y, batch_size), 1):
                            feed = {X: x,Y: np.reshape(y,(len(y),1)),keep_prob:1,initial_state: studio_state}
                            batch_loss,batch_acc, studio_state = sess.run([loss,accuracy, final_state], feed_dict=feed)
                            studio_acc.append(batch_acc)
                            studio_loss.append(batch_loss)
                        print("Studio accuracy: {:.5f}".format(np.mean(studio_acc)))
                        file_writer(fileName,['Epoch= '+str(e),'Studio Acc= '+str(np.mean(studio_acc)) ])
                        file_writer(fileNameLoss,['Epoch= '+str(e),'Studio Loss= '+str(np.mean(studio_loss)) ])
                    #represents index of batch
                    batch_index +=1
                    numberOfBatches = np.maximum(numberOfBatches,ii)
                    count_+=1
                    steps+=1
        hyperpara_index += 1 
        exportModel(X,predictions,hyperpara_index,sess)       
        print('Training Completed')
    
def exportModel(X,predictions,modelVersion,sess):
    export_path_base = sys.argv[-1]
    export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(str(modelVersion)))
    export_model(export_path,X,predictions,sess)    
    print 'Done exporting!'

if __name__ == '__main__':
  tf.app.run()
