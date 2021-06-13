import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os,sys
import pickle
from tqdm import tqdm

def sentences_to_elmo_sentence_embs(messages,batch_size=64):
  sess_config = tf.compat.v1.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  #message_lengths = [len(m.split()) for m in messages]
  module_url = "https://tfhub.dev/google/elmo/2"
  elmo = hub.Module(module_url,trainable=True)
  print("module loaded")
  tf.logging.set_verbosity(tf.logging.ERROR)
  with tf.Session(config=sess_config) as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = []
    for i in tqdm(range(0,len(messages),batch_size)):
      #print("Embedding sentences from {} to {}".format(i,min(i+batch_size,len(messages))-1))
      message_batch = messages[i:i+batch_size]
      #length_batch = message_lengths[i:i+batch_size]
      embeddings_batch = session.run(elmo(message_batch,signature="default",as_dict=True))["default"]
      #embeddings_batch = get_embeddings_list(embeddings_batch, length_batch, ELMO_EMBED_SIZE)
      message_embeddings.extend(embeddings_batch)
  return np.array(message_embeddings)


def load_data_to_numpy(folder="../../data/SMS_SPAM/"):
    #SPAM = 1
    #HAM = 0
    #ABSTAIN = -1
    X = []
    Y = []
    raw = "SMSSpamCollection"
    feat = "sms_embeddings.npy"
    with open(folder+raw, 'r', encoding='latin1') as f:
        for line in f:
            yx = line.split("\t",1)
            if yx[0]=="spam":
                y=1
            else:
                y=0
            x = yx[1]
            X.append(x)
            Y.append(y)
    try:
        X_feats = np.load(folder+feat)
    except:
        print("embeddings are absent in the input folder")
        X_feats=sentences_to_elmo_sentence_embs(X)
    X = np.array(X)
    Y = np.array(Y)
    return X, X_feats, Y

def get_various_data(X, Y, X_feats, temp_len, validation_size = 100, test_size = 200, L_size = 100, U_size = None):
    if U_size == None:
        U_size = X.size - L_size - validation_size - test_size
    index = np.arange(X.size)
    index = np.random.permutation(index)
    X = X[index]
    Y = Y[index]
    X_feats = X_feats[index]

    X_V = X[-validation_size:]
    Y_V = Y[-validation_size:]
    X_feats_V = X_feats[-validation_size:]
    R_V = np.zeros((validation_size, temp_len))

    X_T = X[-(validation_size+test_size):-validation_size]
    Y_T = Y[-(validation_size+test_size):-validation_size]
    X_feats_T = X_feats[-(validation_size+test_size):-validation_size]
    R_T = np.zeros((test_size,temp_len))

    X_L = X[-(validation_size+test_size+L_size):-(validation_size+test_size)]
    Y_L = Y[-(validation_size+test_size+L_size):-(validation_size+test_size)]
    X_feats_L = X_feats[-(validation_size+test_size+L_size):-(validation_size+test_size)]
    R_L = np.zeros((L_size,temp_len))

    # X_U = X[:-(validation_size+test_size+L_size)]
    X_U = X[:U_size]
    X_feats_U = X_feats[:U_size]
    # Y_U = Y[:-(validation_size+test_size+L_size)]
    R_U = np.zeros((U_size,temp_len))

    return X_V,Y_V,X_feats_V,R_V, X_T,Y_T,X_feats_T,R_T, X_L,Y_L,X_feats_L,R_L, X_U,X_feats_U,R_U

def get_test_U_data(X, Y, temp_len, test_size = 200, U_size = None):
    if U_size == None:
        U_size = X.size - test_size
    index = np.arange(X.size)
    index = np.random.permutation(index)
    X = X[index]
    Y = Y[index]

    X_T = X[-(test_size):]
    Y_T = Y[-(test_size):]
    R_T = np.zeros((test_size,temp_len))

    # X_U = X[:-(validation_size+test_size+L_size)]
    X_U = X[:U_size]
    # Y_U = Y[:-(validation_size+test_size+L_size)]
    R_U = np.zeros((U_size,temp_len))

    return X_T,Y_T,R_T, X_U,R_U



