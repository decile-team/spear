import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import random
from .synthesizer import Synthesizer
from .verifier import Verifier

class HeuristicGenerator(object):
    """
    A class to go through the synthesizer-verifier loop
    """

    def __init__(self, train_primitive_matrix, val_primitive_matrix, test_primitive_matrix,
    test_ground, val_ground, train_ground=None, b=0.5):
        """ 
        Initialize HeuristicGenerator object

        b: class prior of most likely class (TODO: use somewhere)
        beta: threshold to decide whether to abstain or label for heuristics
        gamma: threshold to decide whether to call a point vague or not
        """

        self.train_primitive_matrix = train_primitive_matrix
        self.val_primitive_matrix = val_primitive_matrix
        self.val_ground = val_ground
        self.train_ground = train_ground
        self.test_primitive_matrix = test_primitive_matrix
        self.test_ground = test_ground
        
        self.num_classes = len(np.unique(self.train_ground))
        self.b = 1/(self.num_classes)
        print('self.b', self.b)

        self.vf = None
        self.syn = None
        self.hf = []
        self.feat_combos = []
        self.val_lfs = []
        self.train_lfs = []
        self.all_idx = set()
#         self.test_lfs = []

    def apply_heuristics(self, heuristics, primitive_matrix, feat_combos, beta_opt):
        """ 
        Apply given heuristics to given feature matrix X and abstain by beta

        heuristics: list of pre-trained logistic regression models
        feat_combos: primitive indices to apply heuristics to
        beta: best beta value for associated heuristics
        """

        def marginals_to_labels(hf,X,beta):
#             print('X.shape',X.shape)
#             marginals = hf.predict_proba(X)[:,-1]
#             print(marginals.shape)
            prob_classes = hf.predict_proba(X) #Ayush
#             print(self.b + beta)
#             print(prob_classes[0:10])
#             exit()
#             labels_cutoff = np.zeros(prob_classes.shape[0]) #Ayush
#             labels_cutoff = np.zeros(np.shape(marginals)) #normal
#             print('X', X.shape, 'prob_classes.shape',prob_classes.shape)
#             for i in range(prob_classes.shape[0]): #Ayush
            marginals = np.max(prob_classes, axis=1) #Ayush
#             cls = np.argmax(prob_classes[i,:])#+1 #Ayush
            labels_cutoff = np.argmax(prob_classes, axis=1)
            labels_cutoff[labels_cutoff == 0] = -1
            labels_cutoff[np.logical_and((self.b -beta) <= marginals, marginals <= (self.b + beta))] = 0#self.num_classes

#             labels_cutoff[marginals <= (self.b-beta)] = -1. #normal
#             labels_cutoff[marginals >= (self.b+beta)] = 1. #normal
#                 labels_cutoff[marginals >= (self.b+beta)] = cls #Ayush
                
#                 if marginals <= (self.b-beta): #ayush
#                     labels_cutoff[i] = -1
# #                     labels_cutoff[i] = -1. #ayush
#                 if marginals >= (self.b + beta):
#                     if cls != 1:
#                         labels_cutoff[i] = -1
#                     else:
#                         labels_cutoff[i] = cls
                
#                 if marginals >= (self.b+beta):
#                     print('prob_classes[i,:], marginals,cls', prob_classes[i,:], marginals,cls, labels_cutoff[i],i)
#                     exit()
            
#             print(marginals.shape)
#             print('self.b-beta',self.b-beta)
#             print('self.b+beta',self.b+beta)
#             labels_cutoff = np.zeros(np.shape(marginals))
#             labels_cutoff[marginals <= (self.b-beta)] = -1.
#             labels_cutoff[marginals >= (self.b+beta)] = 1.
            return labels_cutoff

        L = np.zeros((np.shape(primitive_matrix)[0],len(heuristics)))
        try:
            uu = []
            for i in feat_combos:
                uu.append(i[0])
#             print('uu', uu)
        except:
            f = feat_combos
            feat_combos = []
            feat_combos.append(f)
        for i,hf in enumerate(heuristics):
            if type(feat_combos[i]) is not tuple:
                feat_combos[i] = (feat_combos[i],)
            L[:,i] = marginals_to_labels(hf,primitive_matrix[:,feat_combos[i]],beta_opt[i])

        self.labels = L
        return L

    def prune_heuristics(self,heuristics,feat_combos,keep=1, mode='normal'):
        """ 
        Selects the best heuristic based on Jaccard Distance and Reliability Metric

        keep: number of heuristics to keep from all generated heuristics
        """

        def calculate_jaccard_distance(num_labeled_total, num_labeled_L):
#             print('num_labeled_total',num_labeled_total)
#             print('num_labeled_L',num_labeled_L)
            scores = np.zeros(np.shape(num_labeled_L)[1])
            for i in range(np.shape(num_labeled_L)[1]):
                scores[i] = np.sum(np.minimum(num_labeled_L[:,i],num_labeled_total))/np.sum(np.maximum(num_labeled_L[:,i],num_labeled_total))
            return 1-scores
        
        L_val = np.array([])
        L_train = np.array([])
        beta_opt = np.array([])
        max_cardinality = len(heuristics)
        for i in range(max_cardinality):
#             print(len(heuristics[i]),'heuristics[i].shape')
#             print(len(feat_combos[i]),'feat_combos[i].shape')
            #Note that the LFs are being applied to the entire val set though they were developed on a subset...
            beta_opt_temp = self.syn.find_optimal_beta(heuristics[i], self.val_primitive_matrix, feat_combos[i], self.val_ground)
            L_temp_val = self.apply_heuristics(heuristics[i], self.val_primitive_matrix, feat_combos[i], beta_opt_temp) 
            self.val_lfs = L_temp_val
            L_temp_train = self.apply_heuristics(heuristics[i], self.train_primitive_matrix, feat_combos[i], beta_opt_temp) 
            self.train_lfs = L_temp_train
            
            L_temp_test = self.apply_heuristics(heuristics[i], self.test_primitive_matrix, feat_combos[i], beta_opt_temp) 
            self.test_lfs = L_temp_test
            
            beta_opt = np.append(beta_opt, beta_opt_temp)
            if i == 0:
                L_val = np.append(L_val, L_temp_val) #converts to 1D array automatically
                L_val = np.reshape(L_val,np.shape(L_temp_val))
                L_train = np.append(L_train, L_temp_train) #converts to 1D array automatically
                L_train = np.reshape(L_train,np.shape(L_temp_train))
            else:
                L_val = np.concatenate((L_val, L_temp_val), axis=1)
                L_train = np.concatenate((L_train, L_temp_train), axis=1)
        
        #Use F1 trade-off for reliability
        acc_cov_scores = [f1_score(self.val_ground, L_val[:,i], average='weighted') for i in range(np.shape(L_val)[1])] 
        acc_cov_scores = np.nan_to_num(acc_cov_scores)
        
        if self.vf != None:
            #Calculate Jaccard score for diversity
            train_num_labeled = np.sum(np.abs(self.vf.L_train.T), axis=0) 
            jaccard_scores = calculate_jaccard_distance(train_num_labeled,np.abs(L_train))
#             print(jaccard_scores)
        else:
            jaccard_scores = np.ones(np.shape(acc_cov_scores))

        #Weighting the two scores to find best heuristic
        combined_scores = 0.5*acc_cov_scores + 0.5*jaccard_scores
        if mode == 'random':
            tmp = np.argsort(combined_scores)[::-1] #ayush
            sort_idx = random.sample(range(0,len(tmp)), keep) #ayush for random
        else:
            sort_idx = np.argsort(combined_scores)[::-1][0:keep]
            # print('max_cardinality', sort_idx)
        
        return sort_idx
     

    def run_synthesizer(self, max_cardinality=1, idx=None, keep=1, model='lr', mode='normal'):
        """ 
        Generates Synthesizer object and saves all generated heuristics

        max_cardinality: max number of features candidate programs take as input
        idx: indices of validation set to fit programs over
        keep: number of heuristics to pass to verifier
        model: train logistic regression ('lr') or decision tree ('dt')
        """
        if idx == None:
            primitive_matrix = self.val_primitive_matrix
            ground = self.val_ground
        else:
            primitive_matrix = self.val_primitive_matrix[idx,:]
            ground = self.val_ground[idx]


        #Generate all possible heuristics
        self.syn = Synthesizer(primitive_matrix, ground, b=self.b)

        #Un-flatten indices
        def index(a, inp):
            i = 0
            remainder = 0
            while inp >= 0:
                remainder = inp
                inp -= len(a[i])
                i+=1
            try:
                return a[i-1][remainder] #TODO: CHECK THIS REMAINDER THING WTF IS HAPPENING
            except:
                import pdb; pdb.set_trace()

        #Select keep best heuristics from generated heuristics
        hf, feat_combos = self.syn.generate_heuristics(model, max_cardinality)
#         print(len(hf[0]), 'length of hf')
        for m in range(max_cardinality):
            if len(self.all_idx) > 0:
                
#                 rmv = list(self.all_idx)
                self.all_idx = list(self.all_idx)
                # print('length of hf', len(hf[m]))
#                 print('hf[m]', hf[m])
#                 print('self.all_idx', self.all_idx)
                
                tmp = np.asarray(self.all_idx)
                rmv = np.where(tmp < len(hf[m]))
    #             print('sort_idx', rmv)
                h = np.delete(hf[m],tmp[rmv])
                hf[m] = []
                for i in h:
                    hf[m].append(i)
                
                # feats = np.delete(feat_combos[m], tmp[rmv])
                # feat_combos[m] = []
                # for i in feats:
                #     feat_combos[m].append(i)

                # print('tmp[rmv] ', (tmp[rmv]))
                if len(tmp[rmv]) > 0:
                    t = np.sort(tmp[rmv])[::-1]
                    for i in t:
                        del feat_combos[m][i]
#                 fcl = np.delete(feat_combos[m], tmp[rmv])
                
#                 print('feat_combos is ', feat_combos[m])
#                 feat_combos[m]= []
#                 if m == 0:
#                     for i in fcl:
#                         feat_combos[m].append((i,))
# #                     feat_combos = feat_combos.tolist()
#                     print(feat_combos[m], 'feat_combos[m]')
#                 else:
#                     feat_combos[m] = fcl
                
                
            
        
#         print(len(hf),'h') 
        sort_idx = self.prune_heuristics(hf,feat_combos, keep, mode)
        self.all_idx = set(self.all_idx)
        for i in sort_idx:
            self.all_idx.add(i)
#         print(len(self.all_idx), 'indices ',self.all_idx)
        for i in sort_idx:
#             print(feat_combos,'feat_combos') 
            self.hf.append(index(hf,i)) 
            self.feat_combos.append(index(feat_combos,i))

        #create appended L matrices for validation and train set
        beta_opt = self.syn.find_optimal_beta(self.hf, self.val_primitive_matrix, self.feat_combos, self.val_ground)
        self.L_val = self.apply_heuristics(self.hf, self.val_primitive_matrix, self.feat_combos, beta_opt)       
        self.L_train = self.apply_heuristics(self.hf, self.train_primitive_matrix, self.feat_combos, beta_opt)  
        self.L_test = self.apply_heuristics(self.hf, self.test_primitive_matrix, self.feat_combos, beta_opt)  
    
    def run_verifier(self):
        """ 
        Generates Verifier object and saves marginals
        """
        ###THIS IS WHERE THE SNORKEL FLAG IS SET!!!!
        self.vf = Verifier(self.L_train, self.L_val, self.val_ground, has_snorkel=False)
        self.vf.train_gen_model()
        self.vf.assign_marginals()

    def gamma_optimizer(self,marginals):
        """ 
        Returns the best gamma parameter for abstain threshold given marginals

        marginals: confidences for data from a single heuristic
        """
        m = len(self.hf)
        gamma = 0.5-(1/(m**(3/2.))) 
        return gamma

    def find_feedback(self):
        """ 
        Finds vague points according to gamma parameter

        self.gamma: confidence past 0.5 that relates to a vague or incorrect point
        """
        #TODO: flag for re-classifying incorrect points
#         incorrect_idx = self.vf.find_incorrect_points(b=self.b)

        gamma_opt = self.gamma_optimizer(self.vf.val_marginals)
        #gamma_opt = self.gamma
        vague_idx = self.vf.find_vague_points(b=self.b, gamma=gamma_opt)
        incorrect_idx = vague_idx
        self.feedback_idx = list(set(list(np.concatenate((vague_idx,incorrect_idx)))))   


    def evaluate(self):
        """ 
        Calculate the accuracy and coverage for train and validation sets
        """
        self.val_marginals = self.vf.val_marginals
        self.train_marginals = self.vf.train_marginals

        def calculate_accuracy(marginals, b, ground):
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.sign(2*(marginals - 0.5))
#             print('ground is ', np.sum(labels == ground), ' labels are ', total)
            return np.sum(labels == ground)/float(total)
    
        def calculate_coverage(marginals, b, ground):
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.sign(2*(marginals - 0.5))
            return total/float(len(labels))

#         print('self.train_marginals.shape', self.train_marginals.shape)
#         print('self.val_marginals.shape', self.val_marginals)
        self.val_accuracy = calculate_accuracy(self.val_marginals, self.b, self.val_ground)
        self.train_accuracy = calculate_accuracy(self.train_marginals, self.b, self.train_ground)
        self.val_coverage = calculate_coverage(self.val_marginals, self.b, self.val_ground)
        self.train_coverage = calculate_coverage(self.train_marginals, self.b, self.train_ground)
        return self.val_accuracy, self.train_accuracy, self.val_coverage, self.train_coverage , self.L_val, self.L_train, self.L_test, self.hf

    def heuristic_stats(self):
        '''For each heuristic, we want the following:
        - idx of the features it relies on
        - if dt, then the thresholds?
        ''' 


        def calculate_accuracy(marginals, b, ground):
            total = np.shape(np.where(marginals != 0.5))[1]
            labels = np.sign(2*(marginals - 0.5))
            return np.sum(labels == ground)/float(total)
    
        def calculate_coverage(marginals, b, ground):
            total = np.shape(np.where(marginals != 0))[1]
            labels = marginals
            return total/float(len(labels))

        stats_table = np.zeros((len(self.hf),7))
        for i in range(len(self.hf)):
            stats_table[i,0] = int(self.feat_combos[i][0])
            try:
                stats_table[i,1] = int(self.feat_combos[i][1])
            except:
                stats_table[i,1] = -1.
            try:
                stats_table[i,2] = int(self.feat_combos[i][2])
            except:
                stats_table[i,2] = -1.    
            
            stats_table[i,3] = calculate_accuracy(self.L_val[:,i], self.b, self.val_ground)
            stats_table[i,4] = calculate_accuracy(self.L_train[:,i], self.b, self.train_ground)
            stats_table[i,5] = calculate_coverage(self.L_val[:,i], self.b, self.val_ground)
            stats_table[i,6] = calculate_coverage(self.L_train[:,i], self.b, self.train_ground)
        
        #Make table
        column_headers = ['Feat 1', 'Feat 2','Feat 3', 'Val Acc', 'Train Acc', 'Val Cov', 'Train Cov']
        pandas_stats_table = pd.DataFrame(stats_table, columns=column_headers)
        return pandas_stats_table


            


