#python classifier_weights.py --n-classes 2 --feats count --max-final-lfs 25 --num-rules 50 --data-dir classifier --filter 0 --dataset imdb
import pickle
import pandas as pd
import numpy as np
import sys
import importlib
from sklearn.linear_model._logistic import LogisticRegression
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import os
import enum
print(os.getcwd())
print(f'system path : {sys.path}')
# from .. import labeling_function, ABSTAIN, preprocessor, continuous_scorer
import LFSet
from ..prelabels import PreLabels
import numpy as np
import re
from greedy_filtering_GraphCut import greedy_lf_generation
from greedy_filtering_updated_conflicts import greedy
import argparse
from submodlib import FacilityLocationFunction
from submodlib import GraphCutFunction
from submodlib import SetCoverFunction


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


class classifierWeights():

    def __init__(self, num_rules=30, feats = 'count' ):
        self.reg_type ='l2'
        self.reg_strength = 1.0
        self.num_rules = num_rules
        self.stepwise_inflation_factor = 1
        if feats =='count':
            self.tokenizer = None
        else:
            self.tokenizer = LemmaTokenizer()
        

    def linear_applier(self, texts, labels, featurize_X, vectorizer, valid_texts=None, valid_labs=None):
        self.featurizer, cv = self.build_ngram_featurizer(texts)
        model = LogisticRegression(
            penalty=self.reg_type, C=self.reg_strength, fit_intercept=False, solver='liblinear', class_weight=None, random_state=25)
        X = self.featurizer(texts)
        # X = featurize_X
        model.fit(X, labels)
        # print('labels are ', labels)
        n_classes = model.coef_.shape[0]
        print(' n_classes ', n_classes)

        name_rulefn_score = []

        if n_classes == 1:
            for idx, weight in enumerate(model.coef_[0]):
                if weight == 0: 
                    continue
                if weight > 0:
                    name_rulefn_score.append( (idx, ClassLabels.HAM.value, weight) )
                else:
                    name_rulefn_score.append( (idx, ClassLabels.SPAM.value, weight) )

        else:
            for class_id in range(n_classes):
                for idx, weight in enumerate(model.coef_[class_id]):
                    if weight <= 0:
                        continue
                    # print(idx, class_id, weight)
                    name_rulefn_score.append( (idx, class_id, weight) )
        # print(name_rulefn_score)
        name_rulefn_score = sorted(name_rulefn_score, key=lambda x: abs(x[-1]), reverse=True)
        name_rulefn_score = name_rulefn_score[:int(self.num_rules * self.stepwise_inflation_factor)]

        self.name_rulefn_score = name_rulefn_score
        self.trained = True

        if self.stepwise_inflation_factor > 1.0:
            rm_idxs = self.stepwise_filter(valid_texts, valid_labs, n_to_delete=(len(name_rulefn_score) - self.num_rules))
            for idx in rm_idxs:
                del self.name_rulefn_score[idx]

        self.rule_strs = {}
        idx2tok = {idx: ngram for ngram, idx in cv.vocabulary_.items()}
        for i, lab, weight in self.name_rulefn_score:
            # print(idx, idx2tok[i], lab, weight)
            # self.rule_strs.append(f'{idx2tok[i]} => {lab}')
            self.rule_strs[idx2tok[i]] = lab

        print(self.rule_strs)
        return self.rule_strs


    def build_ngram_featurizer(self, texts):

        cv = CountVectorizer( tokenizer=self.tokenizer,  ngram_range=(1,2), stop_words='english', min_df=1) # ['a', 'the', ',','.','!']
        cv.fit(texts)

        def featurize(texts):
            corpus_counts = cv.transform(texts)
            valid_feats = np.where(np.sum(corpus_counts, 0)> 2)[1]
            # corpus_counts = corpus_counts[:, valid_feats].toarray()
            corpus_counts = corpus_counts.toarray()
            return corpus_counts

        return featurize, cv

    # enum to hold the class labels


    def returnRules(self):
        ABSTAIN = None
        @preprocessor()
        def convert_to_lower(x):
            return x.lower().strip()
        LFS = []
        # featurize, cv = self.build_ngram_featurizer(texts)
        # X = self.featurizer(texts)

        for cand in self.rule_strs.keys():
            label = ClassLabels(self.rule_strs[cand]) ### same for all
            pattern = cand
            resources = {}
            resources['pattern'] = pattern
            resources['output'] = label
            rule_name = pattern+'_lf'
            @labeling_function(name=rule_name,resources=resources,pre=[convert_to_lower],label=label)
            def f(x,**kwargs):
                result=0
                try:
                    result = re.findall(kwargs["pattern"], x)
                    # print('result is ', result)
                except:
                    print('except ', kwargs["pattern"], x)
                if result:
                    return kwargs["output"]
                else:
                    return ABSTAIN
                
            LFS.append(f)
            
        rules = LFSet("LF")
        rules.add_lf_list(LFS)
        
        return rules, self.featurizer


class ClassLabels(enum.Enum):
    # DESCRIPTION     = 0
    # ENTITY          = 1
    # HUMAN           = 2
    # ABBREVIATION    = 3
    # LOCATION        = 4
    # NUMERIC         = 5

    # zero = 0
    # one = 1
    # two = 2
    # three = 3
    # four = 4
    #five = 5

    SPAM = 0
    HAM = 1

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Inducing LFs through Classifier Weights approach and selecting best subset using greedy')
    parser.add_argument('--data-dir',type=str,help='Directory of dataset')
    parser.add_argument('--n-classes',type=int,help='Number of classes',default=2)
    parser.add_argument('--num-rules',type=int,help='Number of top rules filtered in classifier',default=2) 
    parser.add_argument('--feats',type=str,help='Feature Type (count/lemma)',default='count')
    parser.add_argument('--max-final-lfs',type=int,help='Max number of LFs in greedy approach',default=25) 
    parser.add_argument('--filter', type=int, help='Whether to filter using subset or not', default = 0)
    parser.add_argument('--dataset',type=str,help='Name of the datset')

    args = parser.parse_args()

    dataset = args.dataset
    feats = args.feats
    num_rules = int(args.num_rules) #sys.argv[3]
    num_classes = int(args.n_classes)
    save_dir = args.data_dir #sys.argv[5]
    max_lfs = int(args.max_final_lfs)
    filter_check = int(args.filter)
    if not filter_check:
        num_rules = max_lfs

    print('dataset is ', dataset)
    loader_file = ".labeling.lf_gen.Loaders." + dataset+"_loader"
    # pickle_save = "LFs/"+ dataset + "/" + save_dir
    pickle_save = save_dir
    os.makedirs(pickle_save, exist_ok=True)
    # dl = DataLoader()
    load = importlib.import_module(loader_file)
    dl = load.DataLoader()
    train_primitive_matrix, val_primitive_matrix, test_primitive_matrix,train_ground, val_ground, test_ground, vizer, val_idx, common_idx, train_text, val_text, test_text = dl.load_data(dataset=dataset, split_val = 0.1, feat = feats)

    print('Size of validation set ', len(val_ground))
    print('Size of train set ', len(train_ground))
    print('Size of test set ', len(test_ground))

    print('val_primitive_matrix.shape', val_primitive_matrix.shape)

    # x=[vizer.get_feature_names()[val_idx[i]] for i in common_idx]
    cw = classifierWeights(num_rules, feats)
    rule_label_dict = cw.linear_applier(val_text, val_ground, val_primitive_matrix, vizer)
    rules, featurizer = cw.returnRules()

    train_feats = featurizer(train_text) 
    val_feats = featurizer(val_text)
    test_feats = featurizer(test_text)

    print(' train_feats.shape ' , train_feats.shape)

    


    Y_L = np.zeros((train_feats.shape[0]))
    # print(Y_L.shape)
    imdb_noisy_labels = PreLabels(name="sst5", data=train_text, data_feats = train_feats, 
                                gold_labels=Y_L, rules=rules, labels_enum=ClassLabels, num_classes=num_classes)

    train_L, train_S = imdb_noisy_labels.get_labels()
    train_L[train_L==0] = 100
    train_S[train_S==0] = 100
    train_L[train_L==None] = 0
    train_S[train_S==None] = 0
    train_L[train_L==100] = -1
    train_S[train_S==100] = -1
    print('val_text ', len(val_text))
    print('val_feats ', val_feats.shape)
    # print('val_ground ', (val_ground))
    # print(train_ground)
    imdb_noisy_labels = PreLabels(name="sst5", data=val_text, data_feats = val_feats, 
                                gold_labels=val_ground, rules=rules, labels_enum=ClassLabels, num_classes=num_classes)

    val_L, val_S = imdb_noisy_labels.get_labels()
    val_L[val_L==0] = 100
    val_S[val_S==0] = 100
    val_L[val_L==None] = 0
    val_S[val_S==None] = 0
    val_L[val_L==100] = -1
    val_S[val_S==100] = -1
    
    print(val_L.shape , 'val_L.shape')

    imdb_noisy_labels = PreLabels(name="sst5", data=test_text, data_feats = test_feats, 
                                gold_labels=test_ground, rules=rules, labels_enum=ClassLabels, num_classes=num_classes)

    test_L, test_S = imdb_noisy_labels.get_labels()
    test_L[test_L==0] = 100
    test_S[test_S==0] = 100
    test_L[test_L==None] = 0
    test_S[test_S==None] = 0
    test_L[test_L==100] = -1
    test_S[test_S==100] = -1

    def change_0_to_minus():
        train_L[train_L==0] = 100
        train_S[train_S==0] = 100
        train_L[train_L==-1] = 0
        train_S[train_S==-1] = 0
        train_L[train_L==100] = -1
        train_S[train_S==100] = -1

        val_L[val_L==0] = 100
        val_S[val_S==0] = 100
        val_L[val_L==-1] = 0
        val_S[val_S==-1] = 0
        val_L[val_L==100] = -1
        val_S[val_S==100] = -1
        

        test_L[test_L==0] = 100
        test_S[test_S==0] = 100
        test_L[test_L==None] = 0
        test_S[test_S==None] = 0
        test_L[test_L==100] = -1
        test_S[test_S==100] = -1
    if not filter_check:
        change_0_to_minus() 

    if filter_check:
        val_ground[val_ground==0]=-1
        final_set = greedy_lf_generation(train_L, val_L, val_ground, max_lfs = max_lfs, w=0.5, gamma=0.2)
        print('final set is', final_set)
        lx = list(final_set.values())
        print('lx is ', lx)
        change_0_to_minus()
        val_ground[val_ground==-1]=0


        file_name = 'normal_k.npy'
        np.save(os.path.join(pickle_save, file_name), lx)

        with open(os.path.join(pickle_save, 'generatedLFs.txt'), 'w') as f:
            str_lbls = list(rule_label_dict.keys())
            for i,j in final_set.items():
                f.write(str(str_lbls[int(i)]) +',' + str(j) +'\n')

    
        
        final_idx = list(final_set.keys())
        train_L, train_S = train_L[:,final_idx], train_S[:,final_idx]
        val_L, val_S = val_L[:,final_idx], val_S[:,final_idx]
        test_L, test_S = test_L[:, final_idx], test_S[:, final_idx]
    else:
        file_name = 'normal_k.npy'
        lx = []
        # for i in rule_label_dict.values():
        print('label ', list(rule_label_dict.values()))
        np.save(os.path.join(pickle_save, file_name), list(rule_label_dict.values()))

        with open (os.path.join(pickle_save, 'generatedLFs.txt'), 'w') as f:
            for i,j in rule_label_dict.items():
                f.write(i +',' + str(j) +'\n')

    upto = int(len(val_ground)/2 )

    d_ground, val_ground = val_ground[:upto], val_ground[upto:]
    d_x, val_x = val_feats[:upto], val_feats[upto:]
    d_l, val_l = val_L[:upto,:], val_L[upto:,:]
    d_s, val_s = val_S[:upto,:], val_S[upto:,:]


    def lsnork_to_l_m(lsnork, num_classes):
        m = 1 - np.equal(lsnork, -1).astype(int)
        l = m*lsnork + (1-m)*num_classes
        return l,m

     
    
######## Labeled set ############

    d_d = np.array([1.0] * len(train_feats))
    d_r = np.zeros(d_l.shape)
    d_l, d_m = lsnork_to_l_m(d_l, num_classes)
    
    file_name = 'normal_d_processed.p'
    with open(os.path.join(pickle_save, file_name),"wb") as f:
        pickle.dump(d_x,f)
        pickle.dump(d_l,f)
        pickle.dump(d_m,f)
        pickle.dump(d_ground,f)
        pickle.dump(d_d,f)
        pickle.dump(d_r,f)

######## Validation ############

    val_d = np.array([1.0] * len(train_feats))
    val_r = np.zeros(val_l.shape)
    val_l, val_m = lsnork_to_l_m(val_l, num_classes)
    
    file_name = 'normal_validation_processed.p'
    with open(os.path.join(pickle_save, file_name),"wb") as f:
        pickle.dump(val_x,f)
        pickle.dump(val_l,f)
        pickle.dump(val_m,f)
        pickle.dump(val_ground,f)
        pickle.dump(val_d,f)
        pickle.dump(val_r,f)

######## Unlabeled ############

    U_d = np.array([1.0] * len(train_feats))
    U_r = np.zeros(train_L.shape)
    U_l, U_m = lsnork_to_l_m(train_L, num_classes)
    
    file_name = 'normal_U_processed.p'
    with open(os.path.join(pickle_save, file_name),"wb") as f:
        pickle.dump(train_feats,f)
        pickle.dump(U_l,f)
        pickle.dump(U_m,f)
        pickle.dump(train_ground,f)
        pickle.dump(U_d,f)
        pickle.dump(U_r,f)

######## Test ############

    test_d = np.array([1.0] * len(test_feats))
    test_r = np.zeros(test_L.shape)
    test_l, test_m = lsnork_to_l_m(test_L, num_classes)
    
    file_name = 'normal_test_processed.p'
    with open(os.path.join(pickle_save, file_name),"wb") as f:
        pickle.dump(test_feats,f)
        pickle.dump(test_l,f)
        pickle.dump(test_m,f)
        pickle.dump(test_ground,f)
        pickle.dump(test_d,f)
        pickle.dump(test_r,f)