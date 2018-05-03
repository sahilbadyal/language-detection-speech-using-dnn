import csv
import numpy as np
import re
import tensorflow as tf
import shutil
from tensorflow.contrib import learn

NUMBER_OF_GLOVE_FEATURES = 300
NUMBER_OF_LABELS = 1
def_features = 2 * np.random.random_sample(NUMBER_OF_GLOVE_FEATURES) - 1
def_value = "<None>"
def_value_email = "<email>"
def_value_phone = "<phone>"
MAX_SEQ_LENGTH = 100
DEFAULT_BATCH_SIZE = 100
LEARNING_RATE = 0.001
MODEL_SIGNATURE_NAME = 'predict_sentiment_email'

def tokenize_(s):
    pattern = r'''\d+|[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]|[A-Z][A-Z]+|http[s]?://[\w\./]+|[\w]+@[\w]+\.[\w]+|[a-z][a-z]+|[A-Za-z]\.[\w][\w\.]+|[\w]+|[-'a-z]+|[\S]+'''
    l = re.findall(pattern, s)
    return l

def load_input_and_labels(filename):
    input_list = []
    label_list = []
    cat_list = []   
    with open('./data_in/'+str(filename), 'r') as f:
        next(f)
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            label_list.append(int(row[0]))
            input_list.append(row[1])
            cat_list.append(row[2])
    return input_list,label_list,cat_list


def loadGloVe(filename):
    vocab = []
    embd = []
    file = open('/home/nlp/serving/'+str(filename),'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if (len(row[1:]) == NUMBER_OF_GLOVE_FEATURES):
            vocab.append(row[0])
        features = np.empty(NUMBER_OF_GLOVE_FEATURES,dtype=float)
        i=0
        for each in row[1:]:
            features[i] = float(each)
            i = i+1
        embd.append(features)
        if len(embd) % 100000 == 0:
            print len(embd)
            #break
    vocab.append(def_value)
    vocab.append(def_value_email)
    vocab.append(def_value_phone)
    #def_feature = embd[vocab.index('sachin')]
    #print('Feature = ',def_feature)
    def_feature = 2 * np.random.random_sample(NUMBER_OF_GLOVE_FEATURES) - 1
    embd.append(def_feature)
    embd.append(embd[vocab.index('email')])
    embd.append(embd[vocab.index('number')])
    print (vocab[0])
    print (embd[0])
    print "Loaded GloVe!"
    file.close()
    return vocab,embd

def convert_to_features(input_list,vocab,word_to_int):
    sentence2ints = []
    for each in input_list:
        each = tokenize_(each)
        this_sentence_int = []
        for word in each:
            if word in vocab:
                this_sentence_int.append(word_to_int[word])
            elif '@' in word:
                this_sentence_int.append(word_to_int["<email>"])
            elif word.isdigit() and len(word)==10:
                this_sentence_int.append(word_to_int["<phone>"])
            else:
                this_sentence_int.append(word_to_int["<None>"])
        sentence2ints.append(this_sentence_int)
    
    features = np.zeros((len(sentence2ints), MAX_SEQ_LENGTH), dtype=int)
    for i, row in enumerate(sentence2ints):
        features[i, max((MAX_SEQ_LENGTH-len(row)),0):] = np.array(row[:MAX_SEQ_LENGTH] )
    #print features[:2]
    return features    

def file_writer(file_name,row_):
    with open(file_name, 'a') as f:
        spamwriter = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(row_ )
       
def get_batches(x, y, batch_size=DEFAULT_BATCH_SIZE):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

def get_lookup_features(vocab_size,embedding, embedding_dim,features_train, features_test, features_val):
    #Build Look up table
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    
    X = tf.placeholder(tf.int32, [None, None], name = 'inputs')
    embed = tf.nn.embedding_lookup(W, X)
    #Start Session
    #with tf.device('/cpu:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _,em_train = sess.run([embedding_init, embed], feed_dict={embedding_placeholder:embedding, X:features_train})
        _,em_test = sess.run([embedding_init, embed], feed_dict={embedding_placeholder:embedding, X:features_test})
        _,em_val = sess.run([embedding_init, embed], feed_dict={embedding_placeholder:embedding, X:features_val})
    return em_train,em_test,em_val         

def get_hyperparamters_list(lstm_size_list,lstm_layers_list, batch_size_list, n_epochs_list):
    all_hyperpara_list = []
    for a in lstm_size_list:
        for b in lstm_layers_list:
            for c in batch_size_list:
                for d in n_epochs_list:
                    all_hyperpara_list.append((a,b,c,d))
    return all_hyperpara_list
def get_input_and_label_tensor():
    #Placeholder
    X = tf.placeholder(tf.float32, [None, None, NUMBER_OF_GLOVE_FEATURES], name = 'inputs')
    Y = tf.placeholder(tf.float32, [None, NUMBER_OF_LABELS], name = 'labels')
    one  = tf.constant(1.0)
    keep_prob = tf.placeholder_with_default(one, (), name='keep_prob')
    return X,Y,keep_prob

def build_network(lstm_size,lstm_layers,X,keep_prob):
    #Build Network
    with tf.name_scope('lstm_layer'):	
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        with tf.name_scope('dropout'):
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm]*lstm_layers)
    
        initial_state = cell.zero_state(tf.shape(X)[0] , tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state = initial_state)
    tf.summary.histogram('rnn_activations',outputs)
    with tf.name_scope('affine_layer'):
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1],NUMBER_OF_LABELS, activation_fn=tf.tanh)
    tf.summary.histogram('activations',predictions)
    return lstm,cell,initial_state, outputs, final_state, predictions

def optimize_network(Y,predictions):
    #Optimisation
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(Y - predictions))
    tf.summary.scalar('loss',loss)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    return loss, optimizer

def get_accuracy(Y,predictions):
    #Accuracy
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.float32), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy',accuracy)
    return correct_pred, accuracy

def export_model(export_path,X,predictions,sess):
    try:
        shutil.rmtree(export_path)
    except OSError, e:
        print ("Error: %s - %s." % (e.filename,e.strerror))
    print 'Exporting trained model to', export_path
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(predictions)

    prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'sentence': tensor_info_x},
      outputs={'scores': tensor_info_y},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
      MODEL_SIGNATURE_NAME:
          prediction_signature,
    },
    legacy_init_op=legacy_init_op)

    builder.save()

def getCatWiseData(label,data):
    x = []
    y = []
    for item in data:
        if(item[2]==label):
            x.append(item[0])
            y.append(item[1])
    return x,y
