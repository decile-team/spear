#configurations

import os
import argparse
import sys
from shutil import copyfile

from my_utils import get_list_or_None, None_if_zero, get_list, boolean

def parse_args():
    """
    Func Desc:
    Parse input arguments
    Input: 
    
    Output: 
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--run_mode",default="implication",
            type=str,help="train/test mode",
            metavar='')
    ####################### Output Directory #########################
    parser.add_argument("--output_dir",default="./",
            type=str,help="Output checkpoints, final metrics, summaries in this dir",
            metavar='')

    ######################## Data locations for datafeeder ########################
    parser.add_argument('--data_dir',
            default='./',
            type=str, help='Directory containing data pickles', metavar='')
    parser.add_argument('--d_pickle_name', default='d_processed.p', type=str, metavar='')
    parser.add_argument('--U_pickle_name', default='U_processed.p', type=str, metavar='')
    parser.add_argument('--validation_pickle_name', default='test_processed.p', type=str, metavar='')
    parser.add_argument('--w_infer_out_pickle_name', default='infer_w.p', type=str,
            help='output file name for any inference that was ran on w (rule) network', metavar='')
    parser.add_argument('--f_infer_out_pickle_name', default='infer_f.p', type=str,
            help='output file name for any inference that was ran on f (classification) network', metavar='')

    ######################### Checkpointing #######################################
    parser.add_argument('--num_checkpoints', default=1, type=int,
            help='Number of checkpoints to keep around', metavar='')
    parser.add_argument('--checkpoint_load_mode', default='mru', type=str,
            help='Which kind of checkpoint to restore from. Possible options are '
            'mru: Most recently saved checkpoint. Use this to continue a run'
            'f_d, f_d_U: Use these to load the best checkpoint from these runs', metavar='')

    ######################## Datafeeder configurable params #######################

    parser.add_argument('--num_classes', default=0, type=int, help=
            "Number of classes. If 0, this will be dynamically determined using max of"
            " labels in 'd'.", metavar='')
    # Number of items to load for each data type
    parser.add_argument('--num_load_d', default=0, type=int, help=
            'Number of instances to load from d. If 0 load all', metavar='')
    parser.add_argument('--num_load_U', default=0, type=int, help=
            'Number of instances to load from U. If 0 load all', metavar='')
    parser.add_argument('--num_load_validation', default=0, type=int, help=
            'Number of instances to load from validation. If 0 load all', metavar='')
    parser.add_argument('--shuffle_batches', default=True, type=boolean, help=
	    "Don't shuffle batches. Useful for debugging and stepping through batch by batch", metavar='')    
    parser.add_argument('--min_rule_coverage', default=0, type=int,
	    help="Minimum coverage of a rule in U in order to include it in"
            " co-training. Rules which have coverage less than this are assigned"
            " a constant weight of 1.0", metavar='')
    parser.add_argument('--f_d_class_sampling_str', default='', type=str, help=
            "Comma-separated list of number of times each d instance should be "
            "sampled depending on its class for training f on d. Size of list must equal number of "
            "classes", metavar='')
    parser.add_argument('--rule_classes_str', default='', type=str, help=
            "Comma-separated list of the classes predicted by each rule "
            "if string is empty, rule classes are determined from data associated with rule firings"
            , metavar='')

    ####################### Learning rates ################################
    parser.add_argument('--f_d_adam_lr', default=0.001, type=float, metavar='')
    parser.add_argument('--f_d_U_adam_lr', default=0.01, type=float, metavar='')

    ####################### Batch sizes #############################
    parser.add_argument('--f_d_batch_size', default=100, type=int, metavar='')
    parser.add_argument('--f_d_U_batch_size', default=100, type=int, metavar='')
    parser.add_argument('--test_w_batch_size', default=1000, type=int, metavar='')

    ###################### epochs ######################
    parser.add_argument('--f_d_epochs', default=2, type=int, metavar='')
    parser.add_argument('--f_d_U_epochs', default=2, type=int, metavar='')
    
    ###################### Training Params ##########################
    parser.add_argument('--f_d_metrics_pickle_name', default='metrics_train_f_on_d.p', type=str, metavar='')
    parser.add_argument('--f_d_U_metrics_pickle_name', default='metrics_train_f_on_d_U.p', type=str, metavar='')
    parser.add_argument('--f_d_primary_metric', default='f1_score_1', type=str, help='Metric for best '
            'checkpoint computation. The best metrics pickle will also be stored on this basis.'
            ' Valid values are: accuracy: overall accuracy. f1_score_1: f1_score of class 1.'
            ' avg_f1_score: average of all classes f1_score.',
            metavar='')

    ################ network Parameters #########################
    parser.add_argument('--f_layers_str',default='',type=str,
        help='comma-separated list of number of neurons in each layer of'
            ' the fully-connected f network', metavar='')
    parser.add_argument('--w_layers_str', default='', type=str,
            help='comma-separated list of number of neurons in each layer of'
            ' the fully-connected w network', metavar='')
    parser.add_argument('--dropout_keep_prob', default=0.8, type=float, help='', metavar='')
    parser.add_argument('--network_dropout', default=True, type=boolean,
            help='Use dropout in f and w networks', metavar='')

    ############################## ####################################

    parser.add_argument('--gamma',default=0.1,type=float,help='weighting factor for loss on U'
                                                              'used in implication, pr_loss,' 
                                                              'snorkel, generalized cross entropy etc.',
                                                               metavar='')
    parser.add_argument('--lamda',default=0.1,type=float,help='q for generalized cross entropy'
                                                              'or lr for learning to reweight',
                                                              metavar='')

    parser.add_argument('--early_stopping_p',default=20,type=int,help='early stopping patience (in epochs)')

    parser.add_argument('--use_joint_f_w', default=False, type=boolean, help='whether to utilize w network during inference')


    # Print all args
    print("")
    args = parser.parse_args()
    for arg in sorted(vars(args), key=str.lower):
        print("{} = {}".format(arg, getattr(args, arg)))
    print("")
    
    return args

print("Started Reading Flags")
flags = parse_args()

flags.mode = flags.run_mode
if flags.mode is None:
    raise ValueError('Please provide a run mode')

flags.checkpoint_dir = os.path.join(flags.output_dir, 'checkpoints')
if not os.path.exists(flags.checkpoint_dir):
    os.makedirs(flags.checkpoint_dir)
flags.tensorboard_dir = os.path.join(flags.output_dir, 'tensorboard')
if not os.path.exists(flags.tensorboard_dir):
    os.makedirs(flags.tensorboard_dir)

flags.f_layers = get_list(flags.f_layers_str)
flags.w_layers = get_list(flags.w_layers_str)

flags.f_d_class_sampling = get_list_or_None(flags.f_d_class_sampling_str)
flags.rule_classes = get_list_or_None(flags.rule_classes_str)

# Input pickles
print("Hi1")
flags.d_pickle = os.path.join(flags.output_dir, flags.d_pickle_name)
flags.U_pickle = os.path.join(flags.data_dir, flags.U_pickle_name)
flags.validation_pickle = os.path.join(flags.data_dir, flags.validation_pickle_name)

# Output pickles
flags.w_infer_out_pickle = os.path.join(flags.output_dir, flags.w_infer_out_pickle_name)
flags.f_infer_out_pickle = os.path.join(flags.output_dir, flags.f_infer_out_pickle_name)
flags.f_d_metrics_pickle = os.path.join(flags.output_dir, flags.f_d_metrics_pickle_name)
flags.f_d_U_metrics_pickle = os.path.join(flags.output_dir, flags.f_d_U_metrics_pickle_name)

flags.num_classes = None_if_zero(flags.num_classes)
flags.num_load_d = None_if_zero(flags.num_load_d)
flags.num_load_U = None_if_zero(flags.num_load_U)
flags.num_load_validation = None_if_zero(flags.num_load_validation)
print("Ended Reading Flags")

# Move d pickle to output directory.
d_pickle_orig = os.path.join(flags.data_dir, flags.d_pickle_name)
if os.path.exists(d_pickle_orig):
    copyfile(d_pickle_orig, flags.d_pickle)
else:
    print(flags.d_pickle)
    print("Hi")
    assert os.path.exists(flags.d_pickle)
