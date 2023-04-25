
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import combinations, chain
import math
from statistics import mean
import os
import pickle
from submodlib import FacilityLocationFunction
from submodlib import GraphCutFunction
from submodlib import SetCoverFunction

import warnings
warnings.filterwarnings("ignore")


def greedy_lf_generation(train_metric, val_metric, val_ground, max_lfs, w=3, gamma = 0.3):
    '''
    input:
        train_metric : num_instance x num_LFs matrix
        val_metric : num_instance x num_LFs matrix
        val_ground : gold labels on validation set
        max_lfs : maximum number of LFs to return as subset 

    return : dictionary where key correspond to index of train_L and value is the corresponding label assigned by LF

    Stopping condition is either coverage on labeled set is 100%  or max_lfs gets exhausted.
    ABSTAIN should be equal to 0.
    '''
    len_train_ground = train_metric.shape[0]
    df2=pd.DataFrame(train_metric)
    df4=pd.DataFrame(val_metric)
    accuracy=[]
    for i in range(val_metric.shape[1]):
        correct_label=0
        total_labeled=0
        for j in range(val_metric.shape[0]):
            if val_metric[j][i]==val_ground[j]:
                correct_label+=1
                total_labeled+=1
            elif val_metric[j][i]!=0:
                total_labeled+=1
        if total_labeled>0:
            accuracy.append(correct_label/total_labeled)
        else:
            accuracy.append(0)
    index=list([i for i in range(df2.shape[1])])

    coverage_indi=[]
    for j in list(itertools.combinations(index,1)):
        a=list(j)
        sum1=0
        total = df2[a].sum(axis=1)
        for i in total:
            if i!=0:
                sum1 = sum1+1
        coverage_indi.append(sum1)

    def findworth(score,idxs,w):
        accurac=0
        comb_v=[]
        for i in idxs:
            comb_v.append(score[i])
        accurac=sum(comb_v)
        total=df2[idxs].sum(axis=1)
        total1=df4[idxs].sum(axis=1)
        cov_unlabel=0
        cov_label=0
        for i in list(total):
            if i!=0:
                cov_unlabel+=1
        for j in list(total1):
            if j!=0:
                cov_label+=1
        
        worth=accurac + (w)*(cov_unlabel/len_train_ground)
        return worth , cov_unlabel/len_train_ground, cov_label/len(val_ground), accurac

    def find_agreement(df, idxs):
        conflict = 0
        for i in range(df.shape[0]):
            s = np.array(df.iloc[i, idxs])
            uniques = np.unique(s)
            if len(uniques) > 2:
                conflict+=1
            else:
                pass
        conflicts = conflict / df.shape[0] 
        return 1 - conflicts    

    def Matrix(i,j):
        if i==j:
            return 1
        else:
            return findworth(accuracy, [i,j], w=3)[0] + gamma*find_agreement(df2, [i,j])
    
    s_ij= np.eye(df4.shape[1])
    for i in range(df4.shape[1]):
        for j in range(df4.shape[1]):
                if i != j:
                    s_ij[i][j] = Matrix(i,j)
                else:
                    pass

    s_ij_data= np.zeros((df4.shape[1],2))
    for i in range(df4.shape[1]):
        s_ij_data[i][0]+=accuracy[i]
        #print(coverage_indi[i])
        s_ij_data[i][1]+=coverage_indi[i]
    
    # obj = FacilityLocationFunction(n=df4.shape[1], mode='dense', separate_rep=None, n_rep=None, sijs=None, data=s_ij_data, data_rep=None, metric='cosine', num_neighbors=None, create_dense_cpp_kernel_in_python=True)
    # greedyLis = obj.maximize(budget = 16, optimizer = 'NaiveGreedy',stopIfZeroGain=False, stopIfNegativeGain=False)

    obj = GraphCutFunction(n=df4.shape[1], mode = 'dense', lambdaVal=0.7, separate_rep=False, mgsijs=s_ij, data=None, metric='cosine', num_neighbors=None)
    greedyLis = obj.maximize(budget = max_lfs, optimizer = 'NaiveGreedy',stopIfZeroGain=False, stopIfNegativeGain=False)

    
    best_set = []
    for i in greedyLis:
        best_set.append(i[0])
    print(best_set)

    df6=df2[best_set]

    num_classes = len(np.unique(val_ground))
    
    my_labels=val_metric[:,best_set]
    
    el = list(range(1,num_classes))
    el.append(-1)
    labels_lfs = []
    for i in range(my_labels.shape[1]):
        for j in el:
            if len(np.where(my_labels.T[i]==j)[0]) > 0:
                labels_lfs.append(j)
                break

    print("The best set is", best_set ,"\n Classes of Lfs are", labels_lfs)
    print('len(final_set) ', len(best_set))
    print('len (labels_lfs)', len(labels_lfs))
    final = {}

    assert len(best_set) == len(labels_lfs)
    labels_lfs = [0 if x == -1 else x for x in labels_lfs]
    for i, j in zip(best_set, labels_lfs):
        final[i] = j
    return final

    
