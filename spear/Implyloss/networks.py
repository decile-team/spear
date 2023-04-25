from __future__ import print_function

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import inspect
import json
import os

# from .my_utils import merge_dict_a_into_b

from .utils import merge_dict_a_into_b

w_layers = [512, 512] # comma-separated list of number of neurons in each layer of the fully-connected w network
f_layers = [512, 512] # comma-separated list of number of neurons in each layer of the fully-connected f network
network_dropout = True


def create_initializer(initializer_range=0.02):
    '''
    func desc:
    creates the truncated initializer range of gaussian shape

    Input:
    initializer_range (Defailt = 0.02) - the stddev of the expected range

    Output:
    the truncated normal range 
    '''
    return tf.truncated_normal_initializer(stddev=initializer_range)

def w_network_fully_connected(w_var_scope, num_rules, w_dict, reuse=False, dropout_keep_prob=1.0):
    '''
    Func desc:
    creates the fully connected NN for the w network (rule network)

    Input:
    w_var_scope - the variable scope of the w network
    num_rules - number of rules
    w_dict - rule dictionary
    reuse (default - False)
    dropout_keep_prob (default - 1.0) 

    Output:
    the NN itself with all its layers and weights
    '''
    x = w_dict['x']
    rule = w_dict['rules'] #one hot rule representation
    rules_int = w_dict['rules_int'] #integer rule representation

    rule_or_emb = rule
    
    inputs = tf.concat(values = [x, rule_or_emb], axis=-1)

    prev_layer = inputs

    with tf.variable_scope(w_var_scope, reuse=reuse, initializer=create_initializer()) as vs:
        for i, num_neurons in enumerate(w_layers):
            cur_layer = tf.layers.dense(prev_layer, num_neurons,
                    activation=tf.nn.relu, name='w_layer_%d' % i)
            if network_dropout is True:
                cur_layer = tf.nn.dropout(cur_layer, dropout_keep_prob)
            prev_layer = cur_layer
        logit = tf.layers.dense(prev_layer, 1, name='w_linear_layer')
    return logit

def f_network_fully_connected(f_var_scope, f_dict, num_classes, 
                              reuse=False, ph_vars=None, 
                              dropout_keep_prob=1.0):
    '''
    Func desc:
    creates the fully connected NN for the f network (classification network)

    Input:
    f_var_scope - the variable scope of the f network
    f_dict - rule dictionary
    num_classes - number of classes
    reuse (default - False)
    ph_vars (default - None)
    dropout_keep_prob (default - 1.0) 

    Output:
    the NN itself with all its layers and weights
    '''
    x = f_dict['x']
    if not ph_vars:
        with tf.variable_scope(f_var_scope, reuse=reuse, initializer=create_initializer()) as vs:
            prev_layer = x
            for i, num_neurons in enumerate(f_layers):
                cur_layer = tf.layers.dense(prev_layer, num_neurons,
                        activation=tf.nn.relu)
                if network_dropout is True:
                    cur_layer = tf.nn.dropout(cur_layer, dropout_keep_prob)
                prev_layer = cur_layer

            logits = tf.layers.dense(prev_layer,num_classes)
    else:
        for i in range(0,len(ph_vars)-2,2):
            kernel = ph_vars[i]
            bias = ph_vars[i+1]
            x = tf.matmul(x,kernel)
            x = x + bias
            x = tf.nn.relu(x)
            if network_dropout is True:
                x = tf.nn.dropout(x,dropout_keep_prob)

        kernel = ph_vars[-2]
        bias = ph_vars[-1]
        logits = tf.matmul(x,kernel) + bias
    return logits
