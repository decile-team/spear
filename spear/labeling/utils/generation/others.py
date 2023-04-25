from sklearn.linear_model._logistic import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from ...preprocess import *
from ...lf import *
from ...lf_set import *
from nltk.stem import WordNetLemmatizer
import numpy as np
import enum



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
        # print(' n_classes ', n_classes)

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

        # print(self.rule_strs)
        return self.rule_strs


    def build_ngram_featurizer(self, texts):

        cv = CountVectorizer(tokenizer=self.tokenizer,  ngram_range=(1,2), stop_words='english', min_df=1) # ['a', 'the', ',','.','!']
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
                    # print('except ', kwargs["pattern"], x)
                    pass
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

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]