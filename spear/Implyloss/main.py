# from .my_data_feeders import DataFeeder
# from .my_model import HighLevelSupervisionNetwork
# import .my_networks

from .data_feeders import DataFeeder
from .model import HighLevelSupervisionNetwork
import .networks

import numpy as np
import sys, os, shutil

checkpoint_dir =  './checkpoint'
# data_dir = "/home/parth/Desktop/SEM6/RnD/Learning-From-Rules/data/TREC" # Directory containing data pickles
data_dir = "/home/parth/Desktop/SEM6/RnD/spear/examples/SMS_SPAM/data_pipeline/"
inference_output_dir = './inference_output/'
log_dir = './logs'
metric_pickle_dir = './met_pickl/'
tensorboard_dir =  './tensorboard'



if not os.path.exists(inference_output_dir):
    os.makedirs(inference_output_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(metric_pickle_dir):
    os.makedirs(metric_pickle_dir)

if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

checkpoint_load_mode = 'mru' # Which kind of checkpoint to restore from. Possible options are mru: Most recently saved checkpoint. Use this to continue a run f_d, f_d_U: Use these to load the best checkpoint from these runs 
# d_pickle = data_dir+"d_processed.p"
d_pickle = data_dir+"sms_pickle_L.pkl"
dropout_keep_prob =  0.8
early_stopping_p = 20 # early stopping patience (in epochs)
f_d_adam_lr =  0.0003 # default = 0.01
f_d_batch_size = 16
f_d_class_sampling = [10,10] # Comma-separated list of number of times each d instance should be sampled depending on its class for training f on d. Size of list must equal number of classes.
f_d_epochs = 4 # default = 2
f_d_metrics_pickle = metric_pickle_dir+"metrics_train_f_on_d.p"
f_d_primary_metric = 'accuracy' #'f1_score_1' # Metric for best checkpoint computation. The best metrics pickle will also be stored on this basis. Valid values are: accuracy: overall accuracy. f1_score_1: f1_score of class 1. avg_f1_score: average of all classes f1_score 
f_d_U_adam_lr =  0.0003 # default = 0.01
f_d_U_batch_size = 32
f_d_U_epochs = 4 # default = 2  
f_d_U_metrics_pickle = metric_pickle_dir+"metrics_train_f_on_d_U.p"
f_infer_out_pickle = inference_output_dir+"infer_f.p" # output file name for any inference that was ran on f (classification) network
gamma = 0.1 # weighting factor for loss on U used in implication, pr_loss, snorkel, generalized cross entropy etc. 
lamda = 0.1
min_rule_coverage = 0 # Minimum coverage of a rule in U in order to include it in co-training. Rules which have coverage less than this are assigned a constant weight of 1.0.
mode = "learn2reweight" # "learn2reweight" / "implication" / "pr_loss" / "label_snorkel" / "gcross" / "gcross_snorkel" / "f_d" 
test_mode = "" # "" / test_f" / "test_w" / "test_all"
num_classes = 2 # can be 0. Number of classes. If 0, this will be dynamically determined using max of labels in 'd'.
num_load_d = None # can be 0. Number of instances to load from d. If 0 load all.
num_load_U = None # can be 0. Number of instances to load from U. If 0 load all.
num_load_validation = None # can be 0. Number of instances to load from validation. If 0 load all.
q = "1"
rule_classes = None # Comma-separated list of the classes predicted by each rule if string is empty, rule classes are determined from data associated with rule firings.
shuffle_batches = True # Don't shuffle batches. Useful for debugging and stepping through batch by batch
test_w_batch_size = 1000
# U_pickle = data_dir+"U_processed.p"
U_pickle = data_dir+"sms_pickle_U.pkl"
use_joint_f_w = False # whether to utilize w network during inference
# validation_pickle = data_dir+"validation_processed.p"
validation_pickle = data_dir+"sms_pickle_V.pkl"
w_infer_out_pickle = inference_output_dir+"infer_w.p" # output file name for any inference that was ran on w (rule) network
json_file = data_dir+"sms_json.json"

output_dir = "./" + str(mode) + "_" + str(gamma) + "_" + str(lamda) + "_" + str(q)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if test_mode=="":
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)    
    os.makedirs(checkpoint_dir)

# number of input dir - 1 (data_dir)
# number of output dir - 6 (checkpoint, inference_output, log_dir, metric_pickle, output, tensorboard)





if __name__ == '__main__':
    if(str(test_mode)==""):
        output_text_file=log_dir + "/" + str(mode) + "_" + str(gamma) + "_" + str(lamda) + "_" + str(q)+".txt"
    else:    
        output_text_file=log_dir + "/" + str(test_mode) + "_" + str(mode) + "_" + str(gamma) + "_" + str(lamda) + "_" + str(q)+".txt"
    sys.stdout = open(output_text_file,"w")
    if(test_mode!=""):
        mode = test_mode
    if mode not in ['learn2reweight', 'implication', 'f_d', 'pr_loss', 'gcross',  'label_snorkel', 'pure_snorkel', 'gcross_snorkel', 'test_f', 'test_w', 'test_all']:
        raise ValueError('Invalid run mode ' + mode)

    data_feeder = DataFeeder(d_pickle, 
                             U_pickle, 
                             validation_pickle,
                             json_file,
                             shuffle_batches, 
                             num_load_d, 
                             num_load_U, 
                             num_classes, 
                             f_d_class_sampling, 
                             min_rule_coverage, 
                             rule_classes, 
                             num_load_validation, 
                             f_d_batch_size, 
                             f_d_U_batch_size, 
                             test_w_batch_size,
                             out_dir=output_dir)

    num_features, num_classes, num_rules, num_rules_to_train = data_feeder.get_features_classes_rules()
    print("Number of features: ", num_features)
    print("Number of classes: ",num_classes)
    print("Print num of rules to train: ", num_rules_to_train)
    print("Print num of rules: ", num_rules)
    print("\n\n")
    rule_classes = data_feeder.rule_classes
    w_network = networks.w_network_fully_connected #rule network - CHANGE config in w_network_fully_connected of networks - DONE
    f_network = networks.f_network_fully_connected #classification network - CHANGE config in f_network_fully_connected of networks - DONE
    hls = HighLevelSupervisionNetwork(
            num_features,
            num_classes,
            num_rules,
            num_rules_to_train,
            rule_classes,
            w_network,
            f_network,
            f_d_epochs, 
            f_d_U_epochs, 
            f_d_adam_lr, 
            f_d_U_adam_lr, 
            dropout_keep_prob, 
            f_d_metrics_pickle, 
            f_d_U_metrics_pickle, 
            early_stopping_p, 
            f_d_primary_metric, 
            mode, 
            data_dir, 
            tensorboard_dir, 
            checkpoint_dir, 
            checkpoint_load_mode, 
            gamma, 
            lamda,
            raw_d_x=data_feeder.raw_d.x, #instances from the "d" set
            raw_d_L=data_feeder.raw_d.L) #labels from the "d" set

    # Output 3 digits after decimal point in numpy arrays
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})

    print('Run mode is ' + mode)
    if mode == 'f_d':
        print('training f on d')
        hls.train.train_f_on_d(data_feeder, f_d_epochs)
    elif mode == 'implication':
        print("begin Implication loss training")
        hls.train.train_f_on_d_U(data_feeder, f_d_U_epochs, loss_type='implication')
        print(" Implication loss training end")
    elif mode == 'pr_loss':
        print("begin pr_loss training")
        hls.train.train_f_on_d_U(data_feeder, f_d_U_epochs, loss_type='pr_loss')
        print("pr_loss training end")
    elif mode == 'gcross': # majority_label
        print("gcross")
        hls.train.train_f_on_d_U(data_feeder, f_d_U_epochs, loss_type='gcross')
    elif mode == 'gcross_snorkel':
        print("gcross_snorkel")
        hls.train.train_f_on_d_U(data_feeder, f_d_U_epochs, loss_type='gcross_snorkel')
    elif mode == 'learn2reweight':
        print('learn2reweight')
        hls.train.train_f_on_d_U(data_feeder, f_d_U_epochs, loss_type='learn2reweight')
    elif mode == 'pure_snorkel':
        print("pure_snorkel")
        hls.train.train_f_on_d_U(data_feeder, f_d_U_epochs, loss_type='pure_snorkel')
    elif mode == 'label_snorkel':
        print("label_snorkel")
        hls.train.train_f_on_d_U(data_feeder, f_d_U_epochs, loss_type='label_snorkel')
    elif mode == 'test_f':
        print('Running test_f')
        hls.test.test_f(data_feeder, log_output=True, 
                        save_filename=f_infer_out_pickle, 
                        use_joint_f_w=use_joint_f_w)
    elif mode == 'test_w':
        print('Running test_w')
        hls.test.test_w(data_feeder, log_output=True, save_filename=w_infer_out_pickle+"_test")
    elif mode == 'test_all':
        print('Running all tests')
        print('\ninference on f network ...\n')
        hls.test.test_f(data_feeder, log_output=True, 
                        save_filename=f_infer_out_pickle,
                        use_joint_f_w=use_joint_f_w)
        print('\ninference on w network...')
        print('we only test on instances covered by atleast one rule\n')
        hls.test.test_w(data_feeder, log_output=True, save_filename=w_infer_out_pickle+"_test")
    else:
        assert not "Invalid mode string: %s" % mode

    sys.stdout.close()