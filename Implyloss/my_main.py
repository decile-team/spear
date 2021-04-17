# from my_utils import get_data
# from my_core import Implyloss

# num_classes = 6
# if __name__ == '__main__':
# 	path = "d_processed.p" # need to change this
# 	data = get_data(path) # path will be the path of pickle file
# 	Il = Implyloss(data,num_classes)
# 	Il.optimize()

from my_data_feeders import DataFeeder

from my_model import HighLevelSupervisionNetwork
from my_config import flags as config
import my_networks

import numpy as np


if __name__ == '__main__':
    if config.mode not in ['f_d', 'implication', 'pr_loss',
            'gcross', 'learn2reweight',
            'label_snorkel', 'pure_snorkel', 'gcross_snorkel',
            'test_f', 'test_w', 'test_all']:
        raise ValueError('Invalid run mode ' + config.mode)

    # what is config exactly here ?
    data_feeder = DataFeeder(config.d_pickle, 
                             config.U_pickle, 
                             config.validation_pickle,
                             out_dir=config.output_dir,
                             config=config)

    num_features, num_classes, num_rules, num_rules_to_train = data_feeder.get_features_classes_rules()
    print("Number of features: ", num_features)
    print("Number of classes: ",num_classes)
    print("Print num of rules to train: ", num_rules_to_train)
    print("Print num of rules: ", num_rules)
    print("\n\n")
    rule_classes = data_feeder.rule_classes
    w_network = my_networks.w_network_fully_connected #rule network
    f_network = my_networks.f_network_fully_connected #classification network
    hls = HighLevelSupervisionNetwork(
            num_features,
            num_classes,
            num_rules,
            num_rules_to_train,
            rule_classes,
            w_network,
            f_network,
            raw_d_x=data_feeder.raw_d.x, #instances from the "d" set
            raw_d_L=data_feeder.raw_d.L, #labels from the "d" set
            config=config)

    # Output 3 digits after decimal point in numpy arrays
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})

    mode = config.mode
    print('Run mode is ' + mode)
    if mode == 'f_d':
        print('training f on d')
        hls.train.train_f_on_d(data_feeder, config.f_d_epochs)
    elif mode == 'implication':
        print("begin Implication loss training")
        hls.train.train_f_on_d_U(data_feeder, config.f_d_U_epochs, loss_type='implication')
        print(" Implication loss training end")
    elif mode == 'pr_loss':
        print("begin pr_loss training")
        hls.train.train_f_on_d_U(data_feeder, config.f_d_U_epochs, loss_type='pr_loss')
        print("pr_loss training end")
    elif mode == 'gcross':
        print("gcross")
        hls.train.train_f_on_d_U(data_feeder, config.f_d_U_epochs, loss_type='gcross')
    elif mode == 'gcross_snorkel':
        print("gcross_snorkel")
        hls.train.train_f_on_d_U(data_feeder, config.f_d_U_epochs, loss_type='gcross_snorkel')
    elif mode == 'learn2reweight':
        print('learn2reweight')
        hls.train.train_f_on_d_U(data_feeder, config.f_d_U_epochs, loss_type='learn2reweight')
    elif mode == 'pure_snorkel':
        print("pure_snorkel")
        hls.train.train_f_on_d_U(data_feeder, config.f_d_U_epochs, loss_type='pure_snorkel')
    elif mode == 'label_snorkel':
        print("label_snorkel")
        hls.train.train_f_on_d_U(data_feeder, config.f_d_U_epochs, loss_type='label_snorkel')
    elif mode == 'test_f':
        print('Running test_f')
        hls.test.test_f(data_feeder, log_output=True, 
                        save_filename=config.f_infer_out_pickle, 
                        use_joint_f_w=config.use_joint_f_w)
    elif mode == 'test_w':
        print('Running test_w')
        hls.test.test_w(data_feeder, log_output=True, save_filename=config.w_infer_out_pickle+"_test")
    elif mode == 'test_all':
        print('Running all tests')
        print('\ninference on f network ...\n')
        hls.test.test_f(data_feeder, log_output=True, 
                        save_filename=config.f_infer_out_pickle,
                        use_joint_f_w=config.use_joint_f_w)
        print('\ninference on w network...')
        print('we only test on instances covered by atleast one rule\n')
        hls.test.test_w(data_feeder, log_output=True, save_filename=config.w_infer_out_pickle+"_test")
    else:
        assert not "Invalid mode string: %s" % mode