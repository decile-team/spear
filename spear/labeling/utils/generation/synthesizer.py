import numpy as np
import itertools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """
    def __init__(self, primitive_matrix, val_ground,b=0.5):
        """ 
        Initialize Synthesizer object

        b: class prior of most likely class
        beta: threshold to decide whether to abstain or label for heuristics
        """
        self.val_primitive_matrix = primitive_matrix
        self.val_ground = val_ground
        self.p = np.shape(self.val_primitive_matrix)[1]
        
        self.num_classes = len(np.unique(val_ground))
        self.b = 1/(self.num_classes)
    def generate_feature_combinations(self, cardinality=1):
        """ 
        Create a list of primitive index combinations for given cardinality

        max_cardinality: max number of features each heuristic operates over 
        """
        primitive_idx = range(self.p)
        feature_combinations = []

        for comb in itertools.combinations(primitive_idx, cardinality):
            feature_combinations.append(comb)

        return feature_combinations

    def fit_function(self, comb, model):
        """ 
        Fits a single logistic regression or decision tree model

        comb: feature combination to fit model over
        model: fit logistic regression or a decision tree
        """
        X = self.val_primitive_matrix[:,comb]
        if np.shape(X)[0] == 1:
            X = X.reshape(-1,1)

        # fit decision tree or logistic regression or knn
        if model == 'dt':
#             print(X,' is  len(comb)')
            dt = DecisionTreeClassifier(max_depth=len(comb))
            dt.fit(X,self.val_ground)
            return dt

        elif model == 'lr':
            lr = LogisticRegression()
            lr.fit(X,self.val_ground)
            return lr

        elif model == 'nn':
            nn = KNeighborsClassifier(algorithm='kd_tree')
            nn.fit(X,self.val_ground)
            return nn

    def generate_heuristics(self, model, max_cardinality=1):
        """ 
        Generates heuristics over given feature cardinality

        model: fit logistic regression or a decision tree
        max_cardinality: max number of features each heuristic operates over
        """
        #have to make a dictionary?? or feature combinations here? or list of arrays?
        feature_combinations_final = []
        heuristics_final = []
        for cardinality in range(1, max_cardinality+1):
            feature_combinations = self.generate_feature_combinations(cardinality)
            

            heuristics = []
            for i,comb in enumerate(feature_combinations):
#                 print('feature_combinations', comb)
                heuristics.append(self.fit_function(comb, model))

            feature_combinations_final.append(feature_combinations)
            heuristics_final.append(heuristics)

        return heuristics_final, feature_combinations_final

    def beta_optimizer(self,marginals, ground): # By ayush
#     def beta_optimizer(self,marginals, ground):
        """ 
        Returns the best beta parameter for abstain threshold given marginals
        Uses F1 score that maximizes the F1 score

        marginals: confidences for data from a single heuristic
        """	

        #Set the range of beta params
        #0.25 instead of 0.0 as a min makes controls coverage better
        beta_params = np.linspace(0.25,0.45,10)
#         beta_params = np.linspace(0.05,0.25,5)

        f1 = []		
 		
        for beta in beta_params:		
    #             labels_cutoff = np.zeros(np.shape(marglabels_cutoff[np.where(labels_cutoff == 0)] = -1inals))		
#             print(marginals)
            margin = np.max(marginals, axis=1) #Ayush
            labels_cutoff = np.argmax(marginals, axis=1)
            labels_cutoff[labels_cutoff == 0] = -1
            labels_cutoff[np.logical_and((self.b -beta) <= margin, margin <= (self.b + beta))] = 0#self.num_classes


#             labels_cutoff[marginals <= (self.b-beta)] = -1.		
#             labels_cutoff[marginals >= (self.b+beta)] = 1.
#             print(marginals.shape)
            
#             for j in range(marginals.shape[0]): #Ayush
#                 if marginals[j] <= (self.b-beta): #ayush
#                     labels_cutoff[j] = -1 #ayush
#                 if marginals[j] >= (self.b + beta): #ayush
# #                     labels_cutoff[j] = 1
#                     if cls[j] != 1:
#                         labels_cutoff[j] = -1
#                     else:
#                         labels_cutoff[j] = 0
# #             print(marginals, cls)
            f1.append(f1_score(ground, labels_cutoff, average='weighted'))
        f1 = np.nan_to_num(f1)
        return beta_params[np.argsort(np.array(f1))[-1]]


    def find_optimal_beta(self, heuristics, X, feat_combos, ground):
        """ 
        Returns optimal beta for given heuristics

        heuristics: list of pre-trained logistic regression models
        X: primitive matrix
        feat_combos: feature indices to apply heuristics to
        ground: ground truth associated with X data
        """
#         print('feat_combos in synth', feat_combos)
        try:
            uu = []
            for i in feat_combos:
                uu.append(i[0])
#             print('uu', uu)
        except:
            f = feat_combos
            feat_combos = []
            feat_combos.append(f)
        beta_opt = []
        for i,hf in enumerate(heuristics):
            
#             marginals = hf.predict_proba(X[:,feat_combos[i]]) #normal
            # print('X shape', X.shape, marginals.shape)
            # print(marginals.shape)
#             marginals = marginals[:,-1]  #normal
#             beta_opt.append((self.beta_optimizer(marginals, ground))) #normal
#             if i not in uu:
#                 continue
#             print('feat combos inside loop', type(feat_combos[i]))
            if type(feat_combos[i]) is not tuple:
                feat_combos[i] = (feat_combos[i],)
#             if i ==1:
#                 print('X[:,feat_combos[i]].shape', X[:,feat_combos[i]].shape)
            prob_cls = hf.predict_proba(X[:,feat_combos[i]]) #Ayush
            
#             marginals = np.max(prob_cls, axis=1) #Ayush
#             cls = np.argmax(prob_cls, axis=1) #Ayush
            beta_opt.append((self.beta_optimizer(prob_cls, ground))) # by ayush
        return beta_opt



