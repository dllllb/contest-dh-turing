import json
import gzip
import gensim
from nltk.corpus import stopwords
import numpy as np

def get_words_not_in_w2v():
    with gzip.open('w2v_word_list.json.gz', 'rb') as f:
        w2v_words = set(json.load(f))
    
    def words_not_in_w2v(d):
        alice, bob = [], []
        for post in d['thread']:
            text = post['text']
            if post['userId'] == 'Alice':
                alice.append(len([i for i in text if i.lower() not in w2v_words]))
            else:
                bob.append(len([i for i in text if i.lower() not in w2v_words]))
        if len(alice) == 0:
            alice = [0]
        if len(alice) == 1:
            alice = [1]
        return sum(alice), sum(bob)
    
    return words_not_in_w2v

def wmd(model, stop_words, s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    res = model.wmdistance(s1, s2)
    if np.isfinite(res):
        return res
    else:
        return 0

def get_wmd_dist(vec_path):
    stop_words = set(stopwords.words('english'))
    model = gensim.models.KeyedVectors.load_word2vec_format(vec_path)
    
    def wmd_dist(d):
        alice, bob = [], []
        for post in d['thread']:
            user, text = post['userId'], post['text']
            if user == 'user1':
                alice.append(text)
            else:
                bob.append(text)
        res = wmd(model, stop_words, ' '.join(alice), ' '.join(bob))
        return res, res
    
    return wmd_dist

def get_norm_wmd_dist(vec_path):
    stop_words = set(stopwords.words('english'))
    model = gensim.models.KeyedVectors.load_word2vec_format(vec_path)
    model.init_sims(replace=True)
    
    def norm_wmd_dist(d):
        alice, bob = [], []
        for post in d['thread']:
            user, text = post['userId'], post['text']
            if user == 'user1':
                alice.append(text)
            else:
                bob.append(text)
        res = wmd(model, stop_words, ' '.join(alice), ' '.join(bob))
        return res, res
    
    return norm_wmd_dist

