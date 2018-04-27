import sys
import numpy as np
import sklearn.metrics as metrics
import argparse
import time
import json
import importlib
from models import util as utils

print "==> parsing input arguments"
parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default="ld_covnet_rnn", help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='no commment')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--log_every', type=int, default=100, help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=50000, help='save state every x iteration')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (between 0 and 1)')
parser.add_argument('--no-batch_norm', dest="batch_norm", action='store_false', help='batch normalization')
parser.add_argument('--rnn_num_units', type=int, default=500, help='number of hidden units if the network is RNN')
parser.add_argument('--png_folder', type=str, default="./data/train/png/", help='The folder where spectrograms are placed')
parser.add_argument('--train_file', type=str, default="./data/trainEqual.csv", help='The folder where spectrograms are placed')
parser.add_argument('--val_file', type=str, default="./data/valEqual.csv", help='The folder where spectrograms are placed')

parser.set_defaults(batch_norm=True)

args = parser.parse_args()
args_dict = dict(args._get_kwargs())

print args_dict

train_list = []

with open(args.train_file, "r") as train_listfile:
        train_list = utils.read_csv(train_listfile)

val_list = []

with open(args.val_file, "r") as val_listfile:
        val_list = utils.read_csv(val_listfile)

print "==> %d training examples" % len(train_list)
print "==> %d validation examples" % len(val_list)


args_dict = dict(args._get_kwargs())

args_dict['train_list'] = train_list
args_dict['val_list'] = val_list
    
print "==> using network %s" % args.network

network_module = importlib.import_module("models." + args.network)

network = network_module.Network(**args_dict)


network_name = args.prefix + '%s.bs%d%s%s' % (
    network.name(),
    args.batch_size, 
    ".bn" if args.batch_norm else "", 
    (".d" + str(args.dropout)) if args.dropout>0 else "")
    
print "==> network_name:", network_name


start_epoch = 0
if args.load_state != "":
    start_epoch = network.load_state(args.load_state) + 1

def do_epoch(mode, epoch):
    # mode is 'train' or 'test' or 'predict'
    #print metrics.confusion_matrix(y_true, y_pred)
    
    accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    print "accuracy: %.2f percent" % (accuracy * 100.0 / batches_per_epoch / args.batch_size)
    
    if (mode == "predict"):
        all_prediction = np.vstack(all_prediction)
        pred_filename = "predictions/" + ("equal_split." if args.equal_split else "") + \
                         args.load_state[args.load_state.rfind('/')+1:] + ".csv"
        with open(pred_filename, 'w') as pred_csv:
            for x in all_prediction:
                print >> pred_csv, ",".join([("%.6f" % prob) for prob in x])
                    
    return avg_loss / batches_per_epoch'''


if args.mode == 'train':
    print "==> training"   	
    for epoch in range(start_epoch, args.epochs):
        do_epoch('train', epoch)
        test_loss = do_epoch('test', epoch)
        state_name = 'states/%s.epoch%d.test%.5f.state' % (network_name, epoch, test_loss)
        print "==> saving ... %s" % state_name
        network.save_params(state_name, epoch)
        
elif args.mode == 'test':
    do_epoch('predict', 0)
elif args.mode == 'test_on_train':
    do_epoch('predict_on_train', 0)
else:
    raise Exception("unknown mode")
