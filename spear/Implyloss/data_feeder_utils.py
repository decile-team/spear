import pickle
import numpy as np
import json
#from .config import flags as config

# from .my_data_types import *
from .data_types import *

#reduce_x_features = config.w_network == 'textcnn'
reduce_x_features = False
seq_len = 25

def change_values(l,user_class_to_num_map):
    '''
    Func Desc:
    Replace the class labels in l by sequential labels - 0,1,2,..

    Input:
    l - the class label matrix
    user_class_to_num_map - dictionary storing mapping from original class labels to sequential labels

    Output:
    l - with sequential labels
    '''
    A = l.shape
    d0 = A[0]
    d1 = A[1]
    for i in range(d0):
        for j in range(d1):
            # print(l[i][j])
            l[i][j] = user_class_to_num_map[l[i][j]]
            # print("Hi")
    return l

def load_data(fname, jname, num_load=None):
    '''
    Func Desc:
    load the data from the given file

    Input:
    fname - filename
    num_load (default - None)

    Output:
    the structured F_d_U_Data
    '''
    print('Loading from hoff ', fname)
    with open(fname, 'rb') as f:
        x = pickle.load(f)
        l = pickle.load(f)#.astype(np.int32)
        m = pickle.load(f).astype(np.int32)
        L = pickle.load(f).astype(np.int32)
        d = pickle.load(f).astype(np.int32)
        r = pickle.load(f).astype(np.int32)
        a1 = pickle.load(f)
        a2 = pickle.load(f)
        a3 = pickle.load(f)
        num_classes_pickle = pickle.load(f)#.astype(np.int32)

        # len_x = len(x)
        # assert len(l) == len_x
        # assert len(m) == len_x
        # assert len(L) == len_x
        # assert len(d) == len_x
        # assert len(r) == len_x

        # L = np.reshape(L, (L.shape[0], 1))
        # d = np.reshape(d, (d.shape[0], 1))

        print("batch size", x.shape[0])
        print("num features", x.shape[1])
        print("num classes", num_classes_pickle)
        print("num rules", m.shape[1])

        with open(jname, 'rb') as j:
            enum_map_pickle = json.load(j) # {1->Red, 3->Green, 5->Blue}

        # print(type(enum_map_pickle))
        
        user_class_to_num_map =dict()
        val = 0
        for user_class in enum_map_pickle:
            print(user_class," -> ",val)
            user_class_to_num_map[int(user_class)] = val
            # user_class_to_num_map.add(user_class,val)
            val = val+1
        print("None"," -> ",num_classes_pickle)
        user_class_to_num_map[None] = num_classes_pickle

        print("----------------------------")
        print(user_class_to_num_map)
        print("----------------------------")

        len_x = len(x)
        print("len_x", len_x)
        # print(r)
        # print(m.shape)
        if(r.shape[0]==0):
            r = np.zeros((m.shape))
        print("len_r", len(r))
        print("--------------------------")

        print("Working with l")
        # print(l.shape)
        # print(l)
        # print("Part l1") 
        if(l.shape[0]==0):
        # if l is None:
            print("l is empty")
            l=np.empty(len_x)
            l.fill(None)
        # print(l.shape)
        # print(l)
        # print("Part l2") 
        l = change_values(l,user_class_to_num_map)
        # print(l.shape)
        # print(l)
        # print("Part l3") 
        # l = user_class_to_num_map[l]
        # l = np.vectorize(user_class_to_num_map.get)(l)

        print("--------------------------")  

        print("Working with L")
        # print(L.shape)
        # print(L)
        # print("Part L1")   
        if(L.shape[0]==0):
        # if L is None:
            print("L is empty")
            # L=np.empty(len_x)
            # L.fill(None)
            # L = np.reshape(L, (L.shape[0], 1))
            L = np.full((len_x,1),None)
        # print(L.shape)
        # print(L)
        # print("Part L2")
        L = change_values(L,user_class_to_num_map)
        # print(L.shape)
        # print(L)
        # print("Part L3")
        # L = user_class_to_num_map[L]

        print("--------------------------")

        assert len(l) == len_x
        assert len(m) == len_x
        assert len(L) == len_x
        assert len(d) == len_x
        assert len(r) == len_x
        
        d = np.reshape(d, (d.shape[0], 1))

        if reduce_x_features:
            x = np.concatenate([x[:, 0:seq_len], x[:, 75:(seq_len + 75)],
                x[:, 150:(150 + seq_len)]], axis=-1)

        if num_load is not None and num_load < len_x:
            x = x[:num_load]
            l = l[:num_load]
            m = m[:num_load]
            L = L[:num_load]
            d = d[:num_load]
            r = r[:num_load]

        return F_d_U_Data(x, l, m, L, d, r)


def get_rule_classes(l, num_classes):
    '''
    Func Desc:
    get the different rule_classes 

    Input:
    l ([batch_size, num_rules])
    num_classes (int) - the number of available classes 

    Output:
    rule_classes ([num_rules,1]) - the list of valid classes labelled by rules (say class 2 by r0, class 1 by r1, class 4 by r2 => [2,1,4])
    '''
    # print("rule_class l", l)
    # print(num_classes)
    num_rules = l.shape[1]
    rule_classes = []
    for rule in range(num_rules):
        labels = l[:, rule]
        rule_class = num_classes
        for lbl in labels:
            if lbl != num_classes:
                assert lbl < num_classes
                if rule_class != num_classes:
                    # print("rule", rule)
                    # print("labels", labels)
                    # print("rule_class", rule_class)
                    #print('rule is: ', rule, 'Rule class is: ', rule_class, 'newly found label is: ', lbl, 'num_classes is: ', num_classes)
                    assert(lbl == rule_class)
                else:
                    rule_class = lbl

        if rule_class == num_classes:
            print('No valid label found for rule: ', rule)
            # ok if a rule is just a label (i.e. it does not fire at all)
            # input('Press a key to continue')
        rule_classes.append(rule_class)

    return rule_classes


def extract_rules_satisfying_min_coverage(m, min_coverage):
    '''
    Func Desc:
    extract the rules that satisfy the specified minimum coverage

    Input:
    m ([batch_size, num_rules]) - mij specifies whether ith example is associated with the jth rule
    min_coverage

    Output:
    satisfying_rules - list of satisfying rules
    not_satisfying_rules - list of not satisfying rules
    rule_map_new_to_old
    rule_map_old_to_new 
    '''
    num_rules = len(m[0])
    coverage = np.sum(m, axis=0)
    satisfying_threshold = coverage >= min_coverage
    not_satisfying_threshold = np.logical_not(satisfying_threshold)
    all_rules = np.arange(num_rules)
    satisfying_rules = np.extract(satisfying_threshold, all_rules)
    not_satisfying_rules = np.extract(not_satisfying_threshold, all_rules)

    # Assert that the extraction is stable
    assert np.all(np.sort(satisfying_rules) == satisfying_rules)
    assert np.all(np.sort(not_satisfying_rules) == not_satisfying_rules)

    rule_map_new_to_old = np.concatenate([satisfying_rules,
            not_satisfying_rules])
    rule_map_old_to_new = np.zeros(num_rules, dtype=all_rules.dtype) - 1
    for new, old in enumerate(rule_map_new_to_old):
        rule_map_old_to_new[old] = new

    return satisfying_rules, not_satisfying_rules, rule_map_new_to_old, rule_map_old_to_new


def remap_2d_array(arr, map_old_to_new):
    '''
    Func Desc:
    remap those columns of 2D array that are present in map_old_to_new

    Input:
    arr ([batch_size, num_rules])
    map_old_to_new

    Output:
    modified array

    '''
    old = np.arange(len(map_old_to_new))
    arr[:, old] = arr[:, map_old_to_new]
    return arr


def remap_1d_array(arr, map_old_to_new):
    '''
    Func Desc:
    remap those positions of 1D array that are present in map_old_to_new

    Input:
    arr ([batch_size, num_rules])
    map_old_to_new

    Output:
    modified array
    
    '''
    old = np.arange(len(map_old_to_new))
    arr[old] = arr[map_old_to_new]
    return arr


def modify_d_or_U_using_rule_map(raw_U_or_d, rule_map_old_to_new):
    '''
    Func Desc:
    Modify d or U using the rule map

    Input:
    raw_U_or_d - the raw data (labelled(d) or unlabelled(U))
    rule_map_old_to_new - the rule map

    Output:
    the modified raw_U_or_d

    '''
    remap_2d_array(raw_U_or_d.l, rule_map_old_to_new)
    remap_2d_array(raw_U_or_d.m, rule_map_old_to_new)


def shuffle_F_d_U_Data(data):
    '''
    Func Desc:
    shuffle the input data along the 0th axis i.e. among the different instances 

    Input:
    data

    Output:
    the structured and shuffled F_d_U_Data
    '''
    idx = np.arange(len(data.x))
    np.random.shuffle(idx)
    x = np.take(data.x, idx, axis=0)
    l = np.take(data.l, idx, axis=0)
    m = np.take(data.m, idx, axis=0)
    L = np.take(data.L, idx, axis=0)
    d = np.take(data.d, idx, axis=0)
    r = np.take(data.r, idx, axis=0)

    return F_d_U_Data(x, l, m, L, d, r)


def oversample_f_d(x, labels, sampling_dist):
    '''
    Func Desc:
    Oversample the labelled data using the arguments provided

    Input:
    x ([batch_size, num_features]) - the data
    labels
    samping_dist
    '''
    x_list = []
    L_list = []
    #print('Sampling distribution: ', sampling_dist)
    #print('labels: ', labels[0:4])
    for xx, L in zip(x, labels):
        for i in range(sampling_dist[L]):
            x_list.append(np.array(xx))
            L_list.append(np.array(L))

    return np.array(x_list), np.array(L_list)

def oversample_d(raw_d, sampling_dist):
    '''
    Func Desc:
    performs oversampling on the raw labelled data using the given distribution

    Input:
    raw_d - raw labelled data
    sampling_dist - the given sampling dist

    Output:
    F_d_U_Data
    '''
    x_list = []
    l_list = []
    m_list = []
    L_list = []
    d_list = []
    r_list = []
    #print('Sampling distribution: ', sampling_dist)
    #print('labels: ', raw_d.L[0:4])
    for x, l, m, L, d, r in zip(raw_d.x, raw_d.l, raw_d.m, raw_d.L, raw_d.d, raw_d.r):
        L1 = np.squeeze(L)
        for i in range(sampling_dist[L1]):
            x_list.append(np.array(x))
            l_list.append(np.array(l))
            m_list.append(np.array(m))
            L_list.append(np.array(L))
            d_list.append(np.array(d))
            r_list.append(np.array(r))

    return F_d_U_Data(np.array(x_list),
            np.array(l_list),
            np.array(m_list),
            np.array(L_list),
            np.array(d_list),
            np.array(r_list))