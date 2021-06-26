import sys
sys.path.append('../../')

from spear.labeling import continuous_scorer

from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.keyedvectors import KeyedVectors
import gensim.matutils as gm

print("model loading")
model = None
try:
    model = KeyedVectors.load_word2vec_format('data/SMS_SPAM/glove_w2v.txt', binary=False)
except:
    model = KeyedVectors.load_word2vec_format('../../data/SMS_SPAM/glove_w2v.txt', binary=False)
print("model loaded")

def get_word_vectors(btw_words):
    word_vectors= []
    for word in btw_words:
        try:
            word_vectors.append(model[word])
        except:
            temp = 1
            # store words not avaialble in glove
    return word_vectors

def get_similarity(word_vectors,target_word): # sent(list of word vecs) to word similarity
    similarity = 0
    target_word_vector = 0
    try:
        target_word_vector = model[target_word]
    except:
        # store words not avaialble in glove
        return similarity
    target_word_sparse = gm.any2sparse(target_word_vector,eps=1e-09)
    for wv in word_vectors:
        wv_sparse = gm.any2sparse(wv, eps=1e-09)
        similarity = max(similarity,gm.cossim(wv_sparse,target_word_sparse))
    return similarity

def preprocess(tokens):
    btw_words = [word for word in tokens if word not in STOPWORDS]
    btw_words = [word for word in btw_words if word.isalpha()]
    return btw_words


@continuous_scorer()
def word_similarity(sentence,**kwargs):
    similarity = 0.0
    words = sentence.split()
    words = preprocess(words)
    word_vectors = get_word_vectors(words)
    for w in kwargs['keywords']:
        similarity = min(max(similarity,get_similarity(word_vectors,w)),1.0)

    return similarity

