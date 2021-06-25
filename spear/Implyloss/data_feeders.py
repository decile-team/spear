# used to feed data while training models

import numpy as np
import os
import pickle
import inspect
import time

from spear.Implyloss import *

# from .my_data_types import *
# from .my_data_feeder_utils import *
from .data_types import *
from .data_feeder_utils import *

class DataFeeder():
    def __init__(self, d_pickle, U_pickle, validation_pickle, map_json, 
            shuffle_batches, num_load_d, num_load_U, num_classes, 
            f_d_class_sampling, min_rule_coverage, rule_classes, num_load_validation, 
            f_d_batch_size, f_d_U_batch_size, test_w_batch_size, out_dir='./'):
        '''
        Func Desc:
        Initialize the object with the given parameter files

        Input: 
        self
        d_pickle - labelled data file
        U_pickle - unlabelled data file
        validation_pickle - validation data file
        out_dir (Default = './') - output directory

        Output:
        Void
        '''
        self.f_d_U_start = 0
        self.shuffle_batches=shuffle_batches
        self.out_dir = out_dir
        
        self.raw_d = load_data(d_pickle, map_json, num_load_d)
        self.raw_U = load_data(U_pickle, map_json, num_load_U)

        if num_classes is not None:
            self.num_classes = num_classes
            assert self.num_classes >= np.max(self.raw_d.L) + 1
        else:
            self.num_classes = np.max(self.raw_d.L) + 1

        self.f_d_class_sampling = f_d_class_sampling
        if self.f_d_class_sampling:
            assert len(self.f_d_class_sampling) == self.num_classes
        else:
            self.f_d_class_sampling = [1] * self.num_classes
        # Set f_d_U class sampling dist to be the same as f_d class sampling dist
        self.f_d_U_class_sampling = self.f_d_class_sampling

        self.phi = self.num_classes
        self.num_features = self.raw_d.x.shape[1]
        self.num_rules = self.raw_d.l.shape[1]

        # If min coverage threshold is specified for rules then apply it
        self.min_rule_coverage = min_rule_coverage
        self.num_rules_to_train = self.num_rules
        if self.min_rule_coverage > 0:
            self.satisfying_rules, self.not_satisfying_rules, \
                    self.rule_map_new_to_old, self.rule_map_old_to_new  = \
                    utils.extract_rules_satisfying_min_coverage(self.raw_U.m,
                            self.min_rule_coverage)
            self.num_rules_to_train = len(self.satisfying_rules)
            assert np.all(self.satisfying_rules ==
                    self.rule_map_new_to_old[0:self.num_rules_to_train])
            assert np.all(self.not_satisfying_rules ==
                    self.rule_map_new_to_old[self.num_rules_to_train:])
            print('Originally %d rules. To train on %d rules' %
                    (self.num_rules, self.num_rules_to_train))
            print('Rule map new to old: ', self.rule_map_new_to_old)
            if self.num_rules != self.num_rules_to_train:
                utils.modify_d_or_U_using_rule_map(self.raw_U,
                        self.rule_map_old_to_new)
                utils.modify_d_or_U_using_rule_map(self.raw_d,
                        self.rule_map_old_to_new)

        # Determine rule classes from the truncated rule list
        self.rule_classes = rule_classes 
        if not self.rule_classes:
            self.rule_classes = get_rule_classes(self.raw_d.l, self.num_classes)

        print('Rule classes: ', self.rule_classes)

        # Remove data from U for which no rule makes any predictions
        self.covered_U = self.remove_instances_labeled_by_no_rules(self.raw_U)
        print("length of covered U: {}".format(len(self.covered_U.x)))

        # Now combine d and U data
        self.f_d_U = self.combine_f_d_U(self.raw_d, self.covered_U, self.f_d_U_class_sampling)

        self.f_d = self.convert_raw_d_to_f_d(self.raw_d, num_load=0)

        raw_test_data = load_data(validation_pickle, map_json,
                num_load_validation)

        self.test_f_x, self.test_f_labels, self.test_f_labels_one_hot, \
                self.test_f_l, self.test_f_m, self.test_f_d, self.test_f_r  = \
                self.convert_raw_test_data_to_f(raw_test_data)

        self.test_w = self.convert_raw_test_data_to_w(raw_test_data)
        print('test_w len: ', len(self.test_w.x))

        self.batch_counter = {
                f_d:0,
                f_d_U:0,
                test_w:0,
                }

        self.data_lens = {
                f_d: len(self.f_d.x),
                f_d_U: len(self.f_d_U.x),
                test_w:len(self.test_w.x),
                }

        print('test_w len: ', self.data_lens[test_w])

        self.batch_size = {
                f_d: f_d_batch_size ,
                f_d_U: f_d_U_batch_size,
                test_w: test_w_batch_size,
                }

        self.data_store = {
                test_w: self.test_w,
                }

        self.shuf_indices = {}
        for data_type in [f_d, f_d_U]:
            self.shuf_indices[data_type] = np.arange(self.data_lens[data_type])
            self.reset_batch(data_type)

    
    def convert_raw_test_data_to_f(self, raw_test_data):
        '''
        Func Desc:
        to convert raw test data to f (classification network)

        Input:
        self
        raw_test_data - 

        Output:
        f data with the required parameters
        '''
        x = raw_test_data.x
        labels = np.squeeze(raw_test_data.L)
        assert max(labels) <= self.num_classes - 1 or np.all(labels == self.num_classes)
        labels_one_hot = np.eye(self.num_classes + 1)[labels][:, : -1]
        return x, labels, labels_one_hot, raw_test_data.l, raw_test_data.m, raw_test_data.d, raw_test_data.r

    def convert_raw_test_data_to_w(self, raw_test_data):
        '''
        Func Desc:
        to convert raw test data to w (rule network)

        Input:
        self
        raw_test_data - 

        Output:
        F_d_U_data
        '''
        test_w = self.remove_instances_labeled_by_no_rules(raw_test_data)
        print('Setting value of d to 0 for test data')
        d_new = np.zeros_like(test_w.d)
            
        return F_d_U_Data(test_w.x,
                test_w.l,
                test_w.m,
                test_w.L,
                d_new,
                test_w.r)


    def remove_instances_labeled_by_no_rules(self, raw_U):
        '''
        Func Desc:
        Removes those instances that are labelled by no rules

        Input:
        self
        raw_U - raw Unlabelled Data

        Output:
        F_d_U_data
        '''
        xx = []
        ll = []
        mm = []
        LL = []
        dd = []
        rr = []
        for x, l, m, L, d, r in zip(raw_U.x, raw_U.l, raw_U.m, raw_U.L, raw_U.d, raw_U.r):
            if np.all(l == self.phi):
                continue
            xx.append(x)
            ll.append(l)
            mm.append(m)
            LL.append(L)
            dd.append(d)
            rr.append(r)

        assert len(xx) == len(ll)
        assert len(xx) == len(mm)
        assert len(xx) == len(LL)
        assert len(xx) == len(dd)
        assert len(xx) == len(rr)
        return F_d_U_Data(np.array(xx),
            np.array(ll),
            np.array(mm),
            np.array(LL),
            np.array(dd),
            np.array(rr))


    def combine_f_d_U(self, raw_d, raw_U, d_class_sampling):
        '''
        Func Desc:
        combines the labelled (raw_d) and Unlabelled (raw_U) data

        Input:
        self
        raw_d - labelled data
        raw_U - unlabelled data
        d_class_sampling - sampling distribution
        
        '''
        print('Size of d before oversampling: ', len(raw_d.x))
        print('Size of U (covered) : ', len(raw_U.x))
        # Oversample d according to its class
        #
        # Note that we cannot oversample U according to their true labels since these should not be available
        # during training.
        raw_d = oversample_d(raw_d, d_class_sampling)
        print('Size of d after oversampling: ', len(raw_d.x))

        new_d_d = np.ones_like(raw_d.d)
        new_U_d = np.zeros_like(raw_U.d)

        #num_classes = np.max(raw_d.L) + 1
        #new_U_L = np.zeros_like(raw_U.L) + num_classes
        # We let the true U labels flow through for observation during f_d_U training
        new_U_L = raw_U.L


        xx = np.concatenate((raw_d.x, raw_U.x))
        ll = np.concatenate((raw_d.l, raw_U.l))
        mm = np.concatenate((raw_d.m, raw_U.m))
        LL = np.concatenate((raw_d.L, new_U_L))
        dd = np.concatenate((new_d_d, new_U_d))
        rr = np.concatenate((raw_d.r, raw_U.r))

        print('Size of d_U after combining: ', len(xx))
        f_d_U = F_d_U_Data(xx, ll, mm, LL, dd, rr)
        return f_d_U

    # Need x and true labels only (x, L)
    def convert_raw_d_to_f_d(self, raw_d, num_load=30):
        '''
        Func Desc:
        converts raw d to f

        Input:
        self
        raw_d - 
        num_load (default = 30)

        Output:
        F_d_Data
        '''
        if num_load <= 0:
            num_load = len(raw_d.x)

        print('Loading %d elements from d' % num_load)

        x = raw_d.x[0:num_load]
        label = np.squeeze(raw_d.L[0:num_load])
        x, label = oversample_f_d(x, label, self.f_d_class_sampling)
        print('num instances in d: ', len(x))
        return F_d_Data(x, label)    

    def reset_batch(self, data_type):
        '''
        Func Desc:
        restes the batch

        Input:
        self
        data_type

        Output:

        '''
        self.batch_counter[data_type] = 0
        if not self.shuffle_batches or data_type == test_w:
            print('Not shuffling batch for data type: ', data_type)
            return

        #print('Shuffling batch for data type: ', data_type)
        np.random.shuffle(self.shuf_indices[data_type])

        idx = self.shuf_indices[data_type]
        # We need to actually shuffle each array
        if data_type == f_d:
            np.take(self.f_d.x, idx, axis=0, out=self.f_d.x)
            np.take(self.f_d.labels, idx, axis=0, out=self.f_d.labels)

        elif data_type == f_d_U:
            np.take(self.f_d_U.x, idx, axis=0, out=self.f_d_U.x)
            np.take(self.f_d_U.l, idx, axis=0, out=self.f_d_U.l)
            np.take(self.f_d_U.m, idx, axis=0, out=self.f_d_U.m)
            np.take(self.f_d_U.L, idx, axis=0, out=self.f_d_U.L)
            np.take(self.f_d_U.d, idx, axis=0, out=self.f_d_U.d)
            np.take(self.f_d_U.r, idx, axis=0, out=self.f_d_U.r)
        else:
            raise ValueError('Data type not recognized: ', data_type)


    # Get next batch indices. Shuffle if necessary
    #
    # NOTE: We DO NOT skip the last (incomplete) batch
    def next_batch(self, data_type):
        '''
        Func Desc:
        get the next batch for computation

        Input:
        self
        data_type

        Output:
        start - start of the next batch
        end - end of the next batch

        '''
        batch_size = self.batch_size[data_type]

        num_instances = self.data_lens[data_type]
        total_batch = num_instances // batch_size
        remaining = num_instances % batch_size
        if remaining > 0 and (total_batch == 0 or 'test' in data_type):
            #print('Should not skip last batch')
            skip_last_batch = False
        else:
            #print('Should skip last batch')
            skip_last_batch = True

        if skip_last_batch:
            check = self.batch_counter[data_type] * batch_size + batch_size > self.data_lens[data_type]
        else:
            check = self.batch_counter[data_type] * batch_size >= self.data_lens[data_type]

        #print('check is: ', check)
        if check:
            #print('Resetting batch')
            self.reset_batch(data_type)

        start = self.batch_counter[data_type] * batch_size
        end = min(start + batch_size, self.data_lens[data_type])
        self.batch_counter[data_type] += 1
        return start, end


    # x: [batch_size, num_features]
    # y: [batch_size, num_classes] --> one-hot
    #
    # from d data
    def get_f_d_next_batch(self):
        '''
        Func Desc:
        get the next batch in f_d (labelled data)

        Input:
        self

        Output:
        x ([batch_size, num_features]) - the data
        labels_one_hot
        '''
        if True:
            start, end = self.next_batch(f_d)
        else:
            if np.random.rand() < 0.17:
                self.reset_batch(f_d)
            self.f_d_U_start = (self.f_d_U_start + 1) % 6
            start = self.f_d_U_start
            end = np.random.geometric(0.5)
            end = end + start
            end = min(end, 6)

        x = self.f_d.x[start:end]
        labels = self.f_d.labels[start:end]

        labels_one_hot = np.eye(self.num_classes)[labels]
        return x, labels_one_hot


    def get_f_d_U_next_batch(self):
        '''
        Func Desc:
        get the next batch in f_d_U (labelled + unlabelled data)

        Input:
        self

        Output:
        x ([batch_size, num_features]) - the data
        l ([batch_size, num_rules]) - the data labels
        m ([batch_size, num_rules]) - rule association matrix
        L ([batch_size, 1]) - labelling check vector
        d ([batch_size, 1]) - labelled data check vector
        r ([batch_size, num_rules]) - rule coverage matrix
        '''
        start, end = self.next_batch(f_d_U)

        x = self.f_d_U.x[start:end]
        l = self.f_d_U.l[start:end]
        m = self.f_d_U.m[start:end]
        L = self.f_d_U.L[start:end]
        d = self.f_d_U.d[start:end]
        r = self.f_d_U.r[start:end]
        return x, l, m, L, d, r

    # Number of instances
    def get_f_d_num_instances(self):
        '''
        Func Desc:
        gives the number of data instances in f_d

        Input:
        self

        Output:
        the required count

        '''
        return len(self.f_d.x)

    def get_f_d_U_num_instances(self):
        '''
        Func Desc:
        gives the number of data instances in f_d_U

        Input:
        self

        Output:
        the required count

        '''
        return len(self.f_d_U.x)

    # Batch Sizes
    def get_f_d_batch_size(self):
        '''
        Func Desc:
        gives the batch_size in f_d

        Input:
        self

        Output:
        the required size

        '''
        return self.batch_size[f_d]

    def get_f_d_U_batch_size(self):
        '''
        Func Desc:
        gives the batch_size in f_d_U

        Input:
        self

        Output:
        the required size

        '''
        return self.batch_size[f_d_U]

    def get_batch_size(self, data_type):
        '''
        Func Desc:
        gives the batch_size of the required data type

        Input:
        self
        dat_type

        Output:
        the required size

        '''
        return self.batch_size[data_type]

    def get_batches_per_epoch(self, data_type):
        '''
        Func Desc:
        gives the total number of batches in the required data type

        Input:
        self
        dat_type

        Output:
        the required count

        '''
        num_instances = self.data_lens[data_type]
        batch_size = self.batch_size[data_type]
        total_batch = num_instances // batch_size
        remaining = num_instances % batch_size
        print('num_instances: ', num_instances )
        print('batch_size: ', batch_size )
        print('total_batch: ', total_batch )
        print('remaining: ', remaining )
        # Add last batch if it is the only batch
        # Else last batch is discarded
        #
        # Unless we are in test mode and the entire dataset needs to be tested
        if remaining > 0 and (total_batch == 0 or 'test' in data_type):
            total_batch += 1

        print('total_batch: ', total_batch )
        #print('instances\t batch_size\t total_batch\t remaining')
        #print('%d\t %d\t %d\t %d' % (num_instances, batch_size, total_batch,
        #    remaining))
        return total_batch

    def get_features_classes_rules(self):
        '''
        Func Desc:
        get the features, classes and rules of the object

        Input:
        self

        Output:
        num_features
        num_classes
        num_rules
        num_rules_to_train

        '''
        return self.num_features, self.num_classes, self.num_rules, \
                self.num_rules_to_train

    def get_f_test_data(self, data_type):
        '''
        Func Desc:
        get the test data for f_network

        Input:
        self
        data_type

        Output:
        test_f_x
        test_f_labels_one_hot
        test_f_L
        test_f_m
        test_f_d

        '''
        return self.test_f_x, self.test_f_labels_one_hot, \
                self.test_f_l, self.test_f_m, self.test_f_d

    def get_w_test_data(self, data_type='test_w'):
        '''
        Func Desc:
        get the test data for w_network

        Input:
        self
        data_type (fixed to test_w)

        Output:
        x ([batch_size, num_features]) - the data
        l ([batch_size, num_rules]) - the data labels
        m ([batch_size, num_rules]) - rule association matrix
        L ([batch_size, 1]) - labelling check vector
        d ([batch_size, 1]) - labelled data check vector
        
        '''
        assert data_type  == test_w
        start, end = self.next_batch(data_type)

        x = self.data_store[data_type].x[start:end]
        l = self.data_store[data_type].l[start:end]
        m = self.data_store[data_type].m[start:end]
        L = self.data_store[data_type].L[start:end]
        d = self.data_store[data_type].d[start:end]

        return x, l, m, L, d

