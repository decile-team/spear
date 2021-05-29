import random

import pandas as pd
import numpy as np
import pickle
import re
import os

from obtain_embeddings import sentences_to_elmo_sentence_embs

LABEL_DICT = {"ham": 0, "spam": 1}

def load_data(mode):
    data = pd.read_csv(mode + ".csv", encoding='latin1')
    sentence = [x.lower().strip() for x in data["v2"].tolist()]
    label = [LABEL_DICT[item] for item in data["v1"].tolist()]
    data = list(zip(sentence, label))
    return data


def load_rules():
    rules = []
    file_name = 'rule.txt'
    with open(file_name, 'r', encoding='latin1') as f:
        for line in f:
            list_in = line.strip().split("\t")
            label = LABEL_DICT[list_in[0]]
            pattern = list_in[1]
            sent = list_in[2].lower()
            rules.append((sent, label, pattern))
    return rules


class Generate_data:
    def __init__(self):
        self.num_labels = len(LABEL_DICT)
        self.train_data = load_data("train")
        self.validation_data = load_data("valid")
        self.test_data = load_data("test")
        self.rules = load_rules()
        self.num_rules = len(self.rules)

    def fire_rules(self,sentence):
        #returns m and l values for the sentence
        m = np.zeros(self.num_rules)
        l = self.num_labels + np.zeros(self.num_rules)

        for rid,(sent,lab,pat) in enumerate(self.rules):
            pattern = re.compile(pat)
            rule_label = lab
            result = re.findall(pattern, sentence.strip().lower())                        
            if result:
                m[rid] = 1
                l[rid] = lab
        return m,l


    def _geneate_pickles(self,mode):
        if mode == "U":
            data = self.train_data
        elif mode == "test":
            data = self.test_data
        elif mode == "validation":
            data = self.validation_data
        else:
            print("Error: Wrong mode")
            exit()

        xx = []
        xl = []
        xm = []
        xL = []
        xd = []
        xr = []

        for sentence,label in data:
            xx.append(sentence)
            if mode == "U":
                xL.append(self.num_labels)
            else:
                xL.append(label)
            xd.append(0)
            xr.append(np.zeros(self.num_rules))
            m,l = self.fire_rules(sentence)
            xm.append(m)
            xl.append(l)

        with open("{}_processed.p".format(mode),"wb") as pkl_f, open("{}_sentences.txt".format(mode),"w") as txt_f:
            for sentence in xx:
                txt_f.write(sentence.strip()+'\n')
            print(len(xx))
            xx = sentences_to_elmo_sentence_embs(xx)
            print(len(xx))
            pickle.dump(np.array(xx),pkl_f)
            pickle.dump(np.array(xl),pkl_f)
            pickle.dump(np.array(xm),pkl_f)
            pickle.dump(np.array(xL),pkl_f)
            pickle.dump(np.array(xd),pkl_f)
            pickle.dump(np.array(xr),pkl_f)

    def generate_pickles(self):
        #=== d_processed.p ====#
        d_x = []
        d_l = []
        d_m = []
        d_L = []
        d_d = []
        d_r = []
        for rule_id,(sentence,label,pattern) in enumerate(self.rules):
            if sentence in d_x:
                s_idx = d_x.index(sentence)
                if label == d_L[s_idx]:
                    d_r[s_idx][rule_id]=1
                    continue
            d_x.append(sentence)
            d_d.append(1)
            d_L.append(label)
            m = np.zeros(self.num_rules)
            l = self.num_labels + np.zeros(self.num_rules)
            r = np.zeros(self.num_rules)
            r[rule_id] = 1
            d_r.append(r)
            m,l = self.fire_rules(sentence)
            assert m[rule_id] == 1
            d_m.append(m)
            d_l.append(l)
        with open("d_processed.p", "wb") as pkl_f, open("d_sentences.txt","w") as txt_f:
            for sentence in d_x:
                txt_f.write(sentence.strip()+'\n')
            print(len(d_x))
            d_x = sentences_to_elmo_sentence_embs(d_x)
            print(len(d_x))
            pickle.dump(np.array(d_x), pkl_f)
            pickle.dump(np.array(d_l), pkl_f)
            pickle.dump(np.array(d_m), pkl_f)
            pickle.dump(np.array(d_L), pkl_f)
            pickle.dump(np.array(d_d), pkl_f)
            pickle.dump(np.array(d_r), pkl_f)
        #====== U_processed ======#
        self._geneate_pickles("U")
        self._geneate_pickles("validation")
        self._geneate_pickles("test")



if __name__ == '__main__':
    obj = Generate_data()
    obj.generate_pickles()
