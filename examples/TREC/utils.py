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


def load_data_to_numpy(folder="../../data/TREC/",file_name="train.txt"):
    label_map = {"DESC": "DESCRIPTION",
                "ENTY": "ENTITY",
                "HUM": "HUMAN",
                "ABBR": "ABBREVIATION",
                "LOC": "LOCATION",
                "NUM": "NUMERIC"}
    
    X, Y =  [], []
    data_file = folder+file_name
    with open(data_file, 'r', encoding='latin1') as f:
        for line in f:
            label = label_map[line.split()[0].split(":")[0]]
            if file_name == "test.txt":
                sentence = (" ".join(line.split()[1:]))
                X.append(sentence)
                Y.append(label)
            else:
                sentence = (" ".join(line.split(":")[1:])).lower().strip()
                X.append(sentence)
                Y.append(label)
    
    if file_name == "test.txt":
        feat = "test_trec_embeddings.npy"
    elif file_name == "train.txt":
        feat = "train_trec_embeddings.npy"
    elif file_name == "valid.txt":
        feat = "valid_trec_embeddings.npy"
    feat_file = folder + feat
    try:
        X_feats = np.load(folder+feat)
    except:
        print("embeddings are absent in the input folder")
        X_feats=sentences_to_elmo_sentence_embs(X)

    X = np.array(X)
    Y = np.array(Y)
    return X, X_feats, Y
