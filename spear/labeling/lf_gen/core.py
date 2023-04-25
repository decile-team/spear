from typing import Any, Callable, List, Tuple, Set, Mapping, Optional
import numpy as np
import matplotlib.pyplot as plt
import sys
from ..utils.generation.greedy_filtering_GraphCut import greedy_lf_generation
# from ..utils import labeling_function, ABSTAIN, preprocessor, continuous_scorer
from ..lf import labeling_function, ABSTAIN
from ..preprocess import preprocessor
from ..continuous_scoring import continuous_scorer
from ..utils.generation import HeuristicGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model._logistic import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from ..utils.generation import Synthesizer
from ..lf_set import LFSet
from ..prelabels import PreLabels
import pickle
import os
import re
import enum
import warnings
from sklearn import model_selection as cross_validation
from ..utils.generation.DeepLSTM import *
from ..utils.generation.others import *
warnings.filterwarnings("ignore")
import importlib


class LFgenerator :

    """Generator class for Labelling Functions

    Args:
        dataset (str): dataset for which LFs are to be generated
        model (str): model to use (dt/lr/knn). Defaults to dt
        cardinality (int): Additional resources for the LF. Defaults to 1.
        numloops (int): PThe number of loops to precess for.
        model_feats (str): Features to use (lstm/count/lemma). Defaults to count.
    """

    def __init__(
        self, 
        dataset:str,
        model:str, 
        cardinality:int, 
        numloops:int, 
        model_feats:str) -> None :
        """Instantiates LFgenerator class with the set of inputs      
        """
        self.dataset = dataset
        self.model = model if model else 'dt'
        self.cardinality = cardinality if cardinality else 1
        self.numloops = numloops
        self.model_feats = model_feats if model_feats else 'count'
    

    def __call__(self, dpath:str, savepath:str) -> List[Tuple[str, str]]:
        """Function to generate and save LFs as numpy array firings

        Inputs : 
            dpath : File containing the data
            savepath : Directory to save the generate LFs

        Returns : 
            List of tuples containng the Labling words and their corresponding labels
        """

        print(f'save path : {savepath}')
        print('dataset is ', self.dataset)
        loader_file = ".labeling.lf_gen.Loaders." + self.dataset+"_loader"
        print(os.getcwd())  

        load = importlib.import_module(loader_file, package='spear')
        dl = load.DataLoader()
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground,\
            val_ground, test_ground, vizer, val_idx, common_idx, train_text, val_text, test_text\
            = dl.load_data(dataset=self.dataset, data_path=dpath, split_val = 0.1, feat = self.model_feats)
        self.matrix = (train_primitive_matrix, val_primitive_matrix, test_primitive_matrix)

        print('test length ', len(test_ground))
        print(f'train primitive matrix : {train_primitive_matrix.shape}')

        self.txt = (train_text, val_text, test_text)
        self.gnd = (train_ground, val_ground, test_ground)
        x = [vizer.get_feature_names_out()[val_idx[i]] for i in common_idx ]

        print('Size of validation set ', len(val_ground))
        print('Size of train set ', len(train_ground))
        print('Size of test set ', len(test_ground))

        print('val_primitive_matrix.shape', val_primitive_matrix.shape)

        num_classes = len(np.unique(train_ground))
        overall = {}
        vals=[]
        mode = 'normal'

        # save_path = "LFs/"+ (self.dataset) + "/" + savepath
        self.savepath = savepath
        os.makedirs(self.savepath, exist_ok=True) 
        # save_path = "generated_data/" + dataset #+ "/" + mode
        print('save_path', self.savepath)
        val_file_name = mode + '_val_LFs.npy'
        train_file_name = mode + '_train_LFs.npy'
        test_file_name = mode + '_test_LFs.npy'

        keep_1st=3
        keep_2nd=1
        training_marginals = []
        HF = []

        validation_accuracy = []
        training_accuracy = []
        validation_coverage = []
        training_coverage = []

        idx = None
        hg = HeuristicGenerator(train_primitive_matrix, val_primitive_matrix, test_primitive_matrix,
                        test_ground, val_ground, train_ground, b=0.5)

        for i in range(3,self.numloops):
            if (i-2)%5 == 0:
                print ("Running iteration: ", str(i-2))

            #Repeat synthesize-prune-verify at each iterations
            if i == 3:
                hg.run_synthesizer(max_cardinality= self.cardinality, idx=idx, keep=keep_1st, model=self.model, mode = mode)
            else:
                hg.run_synthesizer(max_cardinality= self.cardinality, idx=idx, keep=keep_2nd, model=self.model, mode = mode)
            hg.run_verifier()

            #Save evaluation metrics
            val_lfs, train_lfs = [], []
            hf = []
            va,ta, vc, tc, val_lfs, train_lfs, test_lfs, hf = hg.evaluate()
            HF = hf
            validation_accuracy.append(va)
            training_accuracy.append(ta)
            training_marginals.append(hg.vf.train_marginals)
            validation_coverage.append(vc)
            training_coverage.append(tc)

            if i==(self.numloops-1):
                np.save(os.path.join(self.savepath ,val_file_name), val_lfs)
                np.save(os.path.join(self.savepath ,train_file_name), train_lfs)
                np.save(os.path.join(self.savepath ,test_file_name), test_lfs)
                print('labels saved') 
            
            hg.find_feedback()
            idx = hg.feedback_idx
            print('Remaining to be labelled ', len(idx))

            if idx == [] :
                np.save(os.path.join(self.savepath ,val_file_name), val_lfs)
                np.save(os.path.join(self.savepath ,train_file_name), train_lfs)
                np.save(os.path.join(self.savepath ,test_file_name), test_lfs)
                print('indexes exhausted... now saving labels')
                break

        print ("Program Synthesis Train Accuracy: ", training_accuracy[-1])
        print ("Program Synthesis Train Coverage: ", training_coverage[-1])
        print ("Program Synthesis Validation Accuracy: ", validation_accuracy[-1])


        trx = np.load(os.path.join(self.savepath ,train_file_name))
        valx = np.load(os.path.join(self.savepath ,val_file_name))
        testx = np.load(os.path.join(self.savepath ,test_file_name))

        yoyo = list(range(1,num_classes))
        yoyo.append(-1)
        labels_lfs = []
        idxs = []
        for i in range(valx.shape[1]):
            for j in yoyo:
                if len(np.where(valx.T[i]==j)[0]) > 1:
                    labels_lfs.append(j)
                    idxs.append(i)
                    break

        trx = trx[:,idxs]
        testx = testx[:,idxs]
        valx = valx[:,idxs]
        print(trx.shape, valx.shape, testx.shape)
        
        

        lx = np.asarray(labels_lfs) 
        lx[np.where(lx==-1)] = 0 
        print('LFS are ', lx)
        file_name = mode + '_k.npy'
        np.save(os.path.join(self.savepath , file_name), lx)

        self.x = (trx, valx, testx, lx)
        np.save(os.path.join(self.savepath,'normal_reef.npy'), training_marginals[-1])

        retinfo = []

        for j, i in zip(lx, hg.heuristic_stats().iloc[:len(idx)]['Feat 1']):
            retinfo.append((str(j), x[int(i)]))
            # f.write(str(j) + ',' + x[int(i)] + '\n')
            # lfwords.append(x[int(i)])
        return retinfo



    def dump(self, retinfo:List[Tuple[str, str]]) -> None :
        """Function to dump all the LF information in pickle files, to be stored in the defined save directory

        Args : 
            retinfo : List of tuples, returned by the call function
        """
      

        (train_text, val_text, test_text) = self.txt
        (train_ground, val_ground, test_ground) = self.gnd
        (trx, valx, testx, lx) = self.x
        (train_primitive_matrix, val_primitive_matrix, test_primitive_matrix) = self.matrix
        num_classes = len(np.unique(train_ground))

        def lsnork_to_l_m(lsnork, num_classes):
            m = 1 - np.equal(lsnork, -1).astype(int)
            l = m*lsnork + (1-m)*num_classes
            return l,m

        if self.model_feats == 'lstm':
            mkt = MakeTokens()
            train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            vocab_size, embedding_vector_length, max_sentence_length =\
            mkt.make(train_text, val_text, test_text)

        upto = int(len(val_ground)/2 ) 
        d_L, U_L = val_ground[:upto], train_ground
        d_x, U_x = val_primitive_matrix[:upto], train_primitive_matrix
        d_l, U_l = valx[:upto,:], trx

        U_text = train_text
        d_text = val_text[:upto]
        val_text = val_text[upto:]
        test_text = test_text

        # pickle_save = "LFs/"+ self.dataset + "/" + savepath
        pickle_save = self.savepath
        def write_txt(name, objs):
            with open(os.path.join(pickle_save , name+'.txt'), 'w') as f:
                for i in objs:
                    f.write(i+'\n')

        write_txt('U', U_text)
        write_txt('d', d_text)
        write_txt('val', val_text)
        write_txt('test', test_text)

        d_d = np.array([1.0] * len(d_x))
        d_r = np.zeros(d_l.shape) 
        d_L[np.where(d_L==-1)[0]] = 0

        d_l[np.where(d_l==-1)]=10
        d_l[np.where(d_l==0)]=-1
        d_l[np.where(d_l==10)]=0
        d_l, d_m = lsnork_to_l_m(d_l, num_classes)


        file_name = 'normal' + '_d_processed.p'
        with open(os.path.join(pickle_save, file_name),"wb") as f:
            pickle.dump(d_x,f)
            pickle.dump(d_l,f)
            pickle.dump(d_m,f)
            pickle.dump(d_L,f)
            pickle.dump(d_d,f)
            pickle.dump(d_r,f)



        U_d = np.array([1.0] * len(U_x))
        U_r = np.zeros(U_l.shape) 

        U_L[np.where(U_L==-1)[0]] = 0

        U_l[np.where(U_l==-1)]=10
        U_l[np.where(U_l==0)]=-1
        U_l[np.where(U_l==10)]=0
        U_l, U_m = lsnork_to_l_m(U_l, num_classes)

        file_name = 'normal' + '_U_processed.p'
        with open(os.path.join(pickle_save, file_name),"wb") as f:
            pickle.dump(U_x,f)
            pickle.dump(U_l,f)
            pickle.dump(U_m,f)
            pickle.dump(U_L,f)
            pickle.dump(U_d,f)
            pickle.dump(U_r,f)



        val_L = val_ground[upto:]
        val_x = val_primitive_matrix[upto:] 
        val_l = valx[upto:,:] 
        val_d = np.array([1.0] * len(val_x))
        val_r = np.zeros(val_l.shape) 
        val_L[np.where(val_L==-1)[0]] = 0

        val_l[np.where(val_l==-1)]=10
        val_l[np.where(val_l==0)]=-1
        val_l[np.where(val_l==10)]=0
        val_l, val_m = lsnork_to_l_m(val_l, num_classes)
        file_name = 'normal' + '_validation_processed.p'
        with open(os.path.join(pickle_save,file_name),"wb") as f:
            pickle.dump(val_x,f)
            pickle.dump(val_l,f)
            pickle.dump(val_m,f)
            pickle.dump(val_L,f)
            pickle.dump(val_d,f)
            pickle.dump(val_r,f)

        test_L = test_ground
        test_x = test_primitive_matrix
        test_l = testx.copy() 
        test_d = np.array([1.0] * len(test_x))
        test_r = np.zeros(test_l.shape) 
        test_L[np.where(test_L==-1)[0]] = 0

        test_l[np.where(test_l==-1)]=10
        test_l[np.where(test_l==0)]=-1
        test_l[np.where(test_l==10)]=0

        test_l, test_m = lsnork_to_l_m(test_l, num_classes)
        file_name = 'normal' + '_test_processed.p'

        with open(os.path.join(pickle_save,file_name),"wb") as f:
            pickle.dump(test_x,f)
            pickle.dump(test_l,f)
            pickle.dump(test_m,f)
            pickle.dump(test_L,f)
            pickle.dump(test_d,f)
            pickle.dump(test_r,f)


        print('Final Size of d set , U set  , validation set , test set', len(d_L), len(U_L), len(val_L), len(test_L))

        lfwords = [el[1] for el in retinfo]

        with open (os.path.join(pickle_save, 'generatedLFs.txt'), 'w') as f:
            for j, i in zip(lx, lfwords):
                f.write(str(j) + ',' + i + '\n')

        print('final LFs are ', lx.shape)
        # return x

    def __repr__(self) :

        return f'dataset {self.dataset}, model {self.model}, card {self.cardinality}, loops {self.numloops}, features {self.model_feats}'



class LFgenerator2():

    """Generator class(2) for Labelling Functions

    Args:
        dataset (str): dataset for which LFs are to be generated
        model (str): model to use (dt/lr/knn). Defaults to dt
        filter (int): Number of filters. Defaults to 0.
        feats(str) : Features to use (lstm/count/lemma). Defaults to count
        numloops (int): Number of loops to precess for.
        max_final_lfs (int): Number of LFs

    """

    def __init__(
        self, 
        dataset:str,
        model:str, 
        filter:int, 
        feats:str,
        numrules:int, 
        max_final_lfs:int) -> None :
        """Instantiates LFgenerator2 class with list of labeling functions      
        """
        self.dataset = dataset
        self.model = model if model else 'dt'
        self.filter = filter if filter else 0
        self.feats = feats if feats else 'count'
        self.numrules = numrules
        self.max_final_lfs = max_final_lfs

    def __call__(self, dpath, savepath) :
        """Function to generate and save LFs as numpy array firings

        Inputs : 
            dpath : File containing the data
            savepath : Directory to save the generate LFs

        Returns : 
            List of tuples containng the Labling words and their corresponding labels
        """

        dataset = self.dataset
        feats = self.feats
        num_rules = int(self.numrules) 
        self.save_dir = savepath 
        max_lfs = int(self.max_final_lfs)
        filter_check = int(self.filter)
        if not filter_check:
            num_rules = max_lfs

        print('dataset is ', dataset)
        loader_file = ".labeling.lf_gen.Loaders." + dataset+"_loader"
        # pickle_save = "LFs/"+ dataset + "/" + save_dir
        pickle_save = self.save_dir
        os.makedirs(pickle_save, exist_ok=True)
        # dl = DataLoader()
        load = importlib.import_module(loader_file, package='spear')
        dl = load.DataLoader()
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, train_ground,\
         val_ground, test_ground, vizer, val_idx, common_idx, train_text, val_text,\
          test_text = dl.load_data(dataset=dataset, data_path=dpath, split_val = 0.1, feat = feats)

        self.matrix = (train_primitive_matrix, val_primitive_matrix, test_primitive_matrix)
        self.gnd = (train_ground, val_ground, test_ground)

        num_classes = len(np.unique(train_ground))
        print('Size of validation set ', len(val_ground))
        print('Size of train set ', len(train_ground))
        print('Size of test set ', len(test_ground))

        print('val_primitive_matrix.shape', val_primitive_matrix.shape)

        cw = classifierWeights(num_rules, feats)
        rule_label_dict = cw.linear_applier(val_text, val_ground, val_primitive_matrix, vizer)
        rules, featurizer = cw.returnRules()

        train_feats = featurizer(train_text) 
        val_feats = featurizer(val_text)
        test_feats = featurizer(test_text)
        self.feats = (train_feats, val_feats, test_feats)

        print(' train_feats.shape ' , train_feats.shape)
        Y_L = np.zeros((train_feats.shape[0]))
        # print(Y_L.shape)
        imdb_noisy_labels = PreLabels(name="prelabels", data=train_text, data_feats = train_feats, 
                                gold_labels=Y_L, rules=rules, labels_enum=ClassLabels, num_classes=num_classes)




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


        train_L, train_S = imdb_noisy_labels.get_labels()
        train_L[train_L==0] = 100
        train_S[train_S==0] = 100
        train_L[train_L==None] = 0
        train_S[train_S==None] = 0
        train_L[train_L==100] = -1
        train_S[train_S==100] = -1
        print('val_text ', len(val_text))
        print('val_feats ', val_feats.shape)

        imdb_noisy_labels = PreLabels(name="prelabels", data=val_text, data_feats = val_feats, 
                                gold_labels=val_ground, rules=rules, labels_enum=ClassLabels, num_classes=num_classes)

        val_L, val_S = imdb_noisy_labels.get_labels()
        val_L[val_L==0] = 100
        val_S[val_S==0] = 100
        val_L[val_L==None] = 0
        val_S[val_S==None] = 0
        val_L[val_L==100] = -1
        val_S[val_S==100] = -1
        
        print(val_L.shape , 'val_L.shape')

        imdb_noisy_labels = PreLabels(name="prelabels", data=test_text, data_feats = test_feats, 
                                    gold_labels=test_ground, rules=rules, labels_enum=ClassLabels, num_classes=num_classes)

        test_L, test_S = imdb_noisy_labels.get_labels()
        test_L[test_L==0] = 100
        test_S[test_S==0] = 100
        test_L[test_L==None] = 0
        test_S[test_S==None] = 0
        test_L[test_L==100] = -1
        test_S[test_S==100] = -1

        self.prelabs = (train_L, train_S, val_L, val_S, test_L, test_S)

        if not filter_check:
            change_0_to_minus() 
        if filter_check :
            val_ground[val_ground==0]=-1
            final_set = greedy_lf_generation(train_L, val_L, val_ground, max_lfs = max_lfs, w=0.5, gamma=0.2)
            print('final set is', final_set)
            lx = list(final_set.values())
            print('lx is ', lx)
            change_0_to_minus()
            val_ground[val_ground==-1]=0

            file_name = 'normal_k.npy'
            np.save(os.path.join(pickle_save, file_name), lx)

            str_lbls = list(rule_label_dict.keys())
            retinfo = []
            for i,j in final_set.items() :
                retinfo.append((str(j), str(str_lbls[int(i)])))
 
            final_idx = list(final_set.keys())
            train_L, train_S = train_L[:,final_idx], train_S[:,final_idx]
            val_L, val_S = val_L[:,final_idx], val_S[:,final_idx]
            test_L, test_S = test_L[:, final_idx], test_S[:, final_idx]
            
        else :
            file_name = 'normal_k.npy'
            lx = []
            # for i in rule_label_dict.values():
            print('label ', list(rule_label_dict.values()))
            np.save(os.path.join(pickle_save, file_name), list(rule_label_dict.values()))

            retinfo = [(str(j), i) for i,j in rule_label_dict.items()]

        # self.rule_label = rule_label_dict
        return retinfo

        



        
    def dump(self, retinfo) :
        """Function to dump all the LF information in pickle files, to be stored in the defined save directory

        Args : 
            retinfo : List of tuples, returned by the call function
        """
        
        (train_ground, val_ground, test_ground) = self.gnd
        (train_feats, val_feats, test_feats) = self.feats
        (train_L, train_S, val_L, val_S, test_L, test_S) = self.prelabs
        (train_primitive_matrix, val_primitive_matrix, test_primitive_matrix) = self.matrix
        num_classes = len(np.unique(train_ground))
        pickle_save = self.save_dir

        upto = int(len(val_ground)/2 )
        d_ground, val_ground = val_ground[:upto], val_ground[upto:]
        d_x, val_x = val_feats[:upto], val_feats[upto:]
        d_l, val_l = val_L[:upto,:], val_L[upto:,:]
        d_s, val_s = val_S[:upto,:], val_S[upto:,:]


        def lsnork_to_l_m(lsnork, num_classes):
            m = 1 - np.equal(lsnork, -1).astype(int)
            l = m*lsnork + (1-m)*num_classes
            return l,m


        ######## TXT FILE GEN ###########

        # str_lbls = list(self.rule_label.keys())
        # if int(self.filter) :
        #     with open(os.path.join(pickle_save, 'generatedLFs.txt'), 'w') as f:
        #         for (i,j) in retinfo:
        #             f.write(str(str_lbls[int(i)]) +',' + str(j) +'\n')
        # else :
        #     with open (os.path.join(pickle_save, 'generatedLFs.txt'), 'w') as f:
        #         for (i,j) in retinfo:
        #             f.write(i +',' + str(j) +'\n')

        with open (os.path.join(pickle_save, 'generatedLFs.txt'), 'w') as f:
            for (i,j) in retinfo:
                f.write(j +',' + i +'\n')


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


    def __repr__(self) :

        return f'dataset {self.dataset}, model {self.model}, filter {self.filter}, feats {self.feats} numRules {self.numrules}, MaxLFs {self.max_final_lfs}'

