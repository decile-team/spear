# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# implementation of generalizer cross entropy loss 
# eq6 of paper https://arxiv.org/pdf/1805.07836.pdf 


# loss for multiclass prediction problem 
# (logits corresponding to a softmax distribution and corresponding labels)
def generalized_cross_entropy(logits, one_hot_labels,q=0.6):
    '''
    Func Desc:
    Computes the generalized cross entropy loss

    Input:
    logits([batch_size, num_classes]) - weights
    one_hot_labels([batch_size, num_classes])
    q (default = 0.6)

    Output:
    loss

    '''
    #logits: [batch_size, num_classes]
    #one_hot_labels: [batch_size, num_classes]
    if q == 0.0:
        #for q=0 in limit, this is usual cross entropy
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,logits=logits)
        loss = tf.reduce_mean(loss)
    else:
        exp_logits = tf.exp(logits)
        normalizer = tf.reduce_sum(exp_logits,axis=-1)#,keep_dims=True)
        normalizer_q = tf.pow(normalizer,q)
        exp_logits_q = tf.exp(logits*q)
        f_j_q = exp_logits_q / normalizer_q
        loss = (1 - f_j_q)/q
        loss = tf.reduce_sum(loss * one_hot_labels, axis=-1)
        loss = tf.reduce_mean(loss)
    return loss

# loss for a particular class
# p = probability of that class 
def generalized_cross_entropy_bernoulli(p,q=0.2):
    '''
    Func Desc:
    computes the bernoulli generalized cross entropy

    Input:
    p - base
    q (default = 0.2) - exponent

    Output:
    loss
    '''
    return (1 - tf.pow(p,q))/q


