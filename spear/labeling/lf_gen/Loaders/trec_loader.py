#!/usr/bin/env python

import numpy as np
import scipy
import json
from sklearn import model_selection as cross_validation
import re
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def parse_file(filename):

    def parse(filename):
        tweet = []
        print(filename)
        with open(filename) as f:
            for line in f:
#                 print(line)
                tweet.append(line)
        return tweet

    tweets = parse(filename)
    gt = []
    plots = []
    idx = []
    for i,twt in enumerate(tweets):
        tweet = twt.split(':')
#         print(tweet)
        genre = tweet[0]
        tweet_txt = tweet[1]
#         tweet_txt = re.sub(r"@\w+","", tweet[1])
#         tweet_txt = ' '.join(tweet_txt.split(' ')[3:])
        
        if 'NUM' in genre:
            plots.append(tweet_txt)
            gt.append(0)
            idx.append(i)
        elif 'LOC' in genre:
            plots.append(tweet_txt)
            gt.append(1)
            idx.append(i)
        elif 'HUM' in genre:
            plots.append(tweet_txt)
            gt.append(2)
            idx.append(i)
        elif 'DESC' in genre:
            plots.append(tweet_txt)
            gt.append(3)
            idx.append(i)
        elif 'ENTY' in genre:
            plots.append(tweet_txt)
            gt.append(4)
            idx.append(i)
        elif 'ABBR' in genre:
            plots.append(tweet_txt)
            gt.append(5)
            idx.append(i)
        else:
            continue  

    print('len of data',len(plots))
    return np.array(plots), np.array(gt)

def split_data(X, plots, y, split_val=0.1):
    np.random.seed(1234)
    num_sample = np.shape(X)[0]
    num_test = 500
    X, plots, y  = shuffle(X, plots, y, random_state = 25)
    X_test = X[0:num_test,:]
    X_train = X[num_test:, :]
    plots_train = plots[num_test:]
    plots_test = plots[0:num_test]

    y_test = y[0:num_test]
    y_train = y[num_test:]

    # split dev/test
    test_ratio = split_val
    X_tr, X_te, y_tr, y_te, plots_tr, plots_te =\
    cross_validation.train_test_split(X_train, y_train, plots_train, test_size = test_ratio, random_state=25)
#     with open('trec_val.txt','w+') as f:
#         for i,j in zip(plots_te, y_te):
# #             x = ji
#             f.write("%s:%s" % (j, i))
              

    return np.array(X_tr.todense()), np.array(X_te.todense()), np.array(X_test.todense()),\
     np.array(y_tr), np.array(y_te), np.array(y_test), plots_tr, plots_te, plots_test


class DataLoader(object):
    """ A class to load in appropriate numpy arrays
    """

    def prune_features(self, val_primitive_matrix, train_primitive_matrix, thresh=0.001):
        val_sum = np.sum(np.abs(val_primitive_matrix),axis=0)
        train_sum = np.sum(np.abs(train_primitive_matrix),axis=0)

        #Only select the indices that fire more than 1% for both datasets
        train_idx = np.where((train_sum >= thresh*np.shape(train_primitive_matrix)[0]))[0]
        val_idx = np.where((val_sum >= thresh*np.shape(val_primitive_matrix)[0]))[0]
        common_idx = list(set(train_idx) & set(val_idx))

        return common_idx

    def load_data(self, dataset, data_path='/home/ayusham/auto_lfs/reef/data/trec/',split_val=0.1, feat = 'count'):
     
        plots, labels = parse_file(data_path+'all.txt')

        def mytokenizer(text):
            return text.split()
        
        #Featurize Plots  
    # niche ki  line original code ka bhag hai
        if feat == 'count':
            vectorizer = CountVectorizer(min_df=1, decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
        elif feat == 'lemma':
            vectorizer = CountVectorizer(min_df=1, binary=True, decode_error='ignore', ngram_range=(1,2) ,\
        tokenizer=LemmaTokenizer(), lowercase = True)
        else:
            vectorizer = CountVectorizer(min_df=1, decode_error='ignore', strip_accents='ascii', ngram_range=(1,2))
        

#         vocab = {'name':0,'name a':1,'how':2,'how does':3,'how to':4,'how can':5,'how should':6,'how would':7,'how could':8,'how will':9,'how do':10,'what':11,'what is':12,'what fastener':13,'how do you':14,'who':15,'who person':16,'who man':17,'who woman':18,'who human':19,'who president':20,'what person':21,'what man':22,'what woman':23,'what human':24,'what president':25,'how much':26,'how many':27,'what kind':28,'what amount':29,'what number':30,'what percentage':31,'capital':32,'capital of':33,'why':34,'why does':35,'why should':36,'why shall':37,'why could':38,'why would':39,'why will':40,'why can':41,'why do':42,'composed':43,'made':44,'composed from':45,'composed through':46,'composed using':47,'composed by':48,'composed of':49,'made from':50,'made through':51,'made using':52,'made by':53,'made of':54,'where':55,'which':56,'where island':57,'which island':58,'what island':59,'who owner':60,'who leads':61,'who governs':62,'who pays':63,'who owns':64,'what is tetrinet':65,'who found':66,'who discovered':67,'who made':68,'who built':69,'who build':70,'who invented':71,'why doesn':72,'used':73,'used for':74,'when':75,'when did':76,'when do':77,'when does':78,'when was':79,'how old':80,'how far':81,'how long':82,'how tall':83,'how wide':84,'how short':85,'how small':86,'how close':87,'fear':88,'fear of':89,'explain':90,'describe':91,'explain can':92,'describe can':93,'who worked':94,'who lived':95,'who guarded':96,'who watched':97,'who played':98,'who ate':99,'who slept':100,'who portrayed':101,'who served':102,'what part':103,'what division':104,'what ratio':105,'who is':106,'who will':107,'who was':108,'what do':109,'what does':110,'enumerate the':111,'list out the':112,'name the':113,'enumerate the various':114,'list out the various':115,'name the various':116,'at which':117,'at how many':118,'at what':119,'in which':120,'in how many':121,'in what':122,'at which age':123,'at which year':124,'at how many age':125,'at how many year':126,'at what age':127,'at what year':128,'in which age':129,'in which year':130,'in how many age':131,'in how many year':132,'in what age':133,'in what year':134,'which play':135,'which game':136,'which movie':137,'which book':138,'what play':139,'what game':140,'what movie':141,'what book':142,'which is':143,'which will':144,'which are':145,'which was':146,'who are':147,'by how':148,'by how much':149,'by how many':150,'where was':151,'where is':152,'studied':153,'patent':154,'man':155,'woman':156,'human':157,'person':158,'stand':159,'mean':160,'meant':161,'called':162,'unusual':163,'origin':164,'country':165,'queen':166,'king':167,'year':168,'novel':169,'speed':170,'abbreviation':171,'percentage':172,'share':173,'number':174,'population':175,'located':176,'thing':177,'instance':178,'object':179,'demands':180,'take':181,'leader':182,'citizen':183,'captain':184,'nationalist':185,'hero':186,'actor':187,'actress':188,'star':189,'gamer':190,'player':191,'lawyer':192,'president':193,'lives':194,'latitude':195,'longitude':196,'alias':197,'nicknamed':198}
        
#         vocab = {'abbreviation':0,'actor':1,'actress':2,'address':3,'age':4,'alias':5,'amount':6,'are':7,'around':8,'at':9,'ate':10,'book':11,'build':12,'built':13,'by':14,'called':15,'can':16,'capital':17,'captain':18,'citizen':19,'close':20,'company':21,'composed':22,'could':23,'country':24,'date':25,'day':26,'demands':27,'describe':28,'did':29,'discovered':30,'division':31,'do':32,'doctor':33,'does':34,'does ':35,'doesn':36,'engineer':37,'enumerate':38,'explain':39,'far':40,'fastener':41,'fastener ':42,'fear':43,'for':44,'found':45,'from':46,'game':47,'gamer':48,'governs':49,'group':50,'groups':51,'guarded':52,'hero':53,'hours':54,'how':55,'human':56,'hypertension':57,'in':58,'instance':59,'invented':60,'is':61,'is ':62,'island':63,'kind':64,'king':65,'latitude':66,'latitude ':67,'lawyer':68,'leader':69,'leads':70,'list':71,'lived':72,'lives':73,'located':74,'long':75,'longitude':76,'made':77,'man':78,'many':79,'mean':80,'meant':81,'minute':82,'model':83,'month':84,'movie':85,'much':86,'name':87,'name ':88,'nationalist':89,'near':90,'nicknamed':91,'novel':92,'number':93,'object':94,'of':95,'old':96,'organization':97,'origin':98,'out':99,'owner':100,'owns':101,'part':102,'patent':103,'pays':104,'percentage':105,'person':106,'play':107,'played':108,'player':109,'poet':110,'population':111,'portrayed':112,'president':113,'queen':114,'ratio':115,'run':116,'seconds':117,'served':118,'shall':119,'share':120,'short':121,'should':122,'should ':123,'situated':124,'slept':125,'small':126,'speed':127,'stand':128,'star':129,'studied':130,'study ':131,'surname':132,'surrounds':133,'take':134,'tall':135,'team':136,'teams':137,'tetrinet':138,'the':139,'thing':140,'through':141,'time':142,'to':143,'trust':144,'unusual':145,'used':146,'using':147,'various':148,'was':149,'was ':150,'watched':151,'what':152,'what ':153,'when':154,'where':155,'where ':156,'which':157,'who':158,'who ':159,'why':160,'wide':161,'will':162,'woman':163,'worked':164,'would':165,'year':166,'you':167}
        # vocab = ["what is","does the","what 's","mean ?","are the","stand for","what does","for ?","in ?","other What","was the","what was","do you","does a","in the","the most","the first","is the","on a","did the","name ?","of the","name the","can i","is a","in what","by the","chancellor of","of ?","were the","from the","into the","What are","australia ?","book to","call the","do italians","italians call","the tallest","to help","was hitler","has the"]
        # vectorizer = CountVectorizer(vocabulary=vocab, tokenizer = mytokenizer, ngram_range=(1,2))#, stop_words='english')
        
        X = vectorizer.fit_transform(plots)

        valid_feats = np.where(np.sum(X,0)> 2)[1]
        X = X[:,valid_feats]
        print(len(valid_feats), valid_feats)

#         Split Dataset into Train, Val, Test
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
        train_ground, val_ground, test_ground,\
        train_plots, val_plots, test_plots = split_data(X, plots, labels, split_val)
        common_idx = []
        #Prune Feature Space
        # common_idx = self.prune_features(val_primitive_matrix, train_primitive_matrix)
        # print('common_idx',len(common_idx))
#         return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx],             np.array(train_ground), np.array(val_ground), np.array(test_ground),train_plots, val_plots, test_plots
        return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
                np.array(train_ground), np.array(val_ground), np.array(test_ground), vectorizer, valid_feats, common_idx, \
            train_plots, val_plots, test_plots

        # return train_primitive_matrix[:,common_idx], val_primitive_matrix[:,common_idx], test_primitive_matrix[:,common_idx], \
        #         np.array(train_ground), np.array(val_ground), np.array(test_ground), vectorizer, valid_feats, common_idx, \
        #     train_plots, val_plots, test_plots


