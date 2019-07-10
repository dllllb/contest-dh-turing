import json
import pandas as pd
from nltk.util import ngrams
import numpy as np
from fuzzywuzzy import fuzz
import re
from nltk.corpus import stopwords

def min_len(d):
    alice, bob = [0], [0]
    for post in d['thread']:
        text = post['text']
        if post['userId'] == 'user1':
            alice.append(len(text))
        else:
            bob.append(len(text))
    return np.min(alice), np.min(bob)


def max_len(d):
    alice, bob = [0], [0]
    for post in d['thread']:
        text = post['text']
        if post['userId'] == 'user1':
            alice.append(len(text))
        else:
            bob.append(len(text))
    return np.max(alice), np.max(bob)


def avg_len(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(len(text))
        else:
            bob.append(len(text))
    if len(alice) == 0:
        alice.append(0)
    if len(bob) == 0:
        bob.append(0)
    return np.mean(alice), np.mean(bob)


# Number of words
#
def min_words(d):
    alice, bob = [0], [0]
    for post in d['thread']:
        user, text = post['userId'], post['text'].split()
        if user == 'user1':
            alice.append(len(text))
        else:
            bob.append(len(text))
    return np.min(alice), np.min(bob)


def max_words(d):
    alice, bob = [0], [0]
    for post in d['thread']:
        user, text = post['userId'], post['text'].split()
        if user == 'user1':
            alice.append(len(text))
        else:
            bob.append(len(text))
    return np.max(alice), np.max(bob)


def avg_words(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text'].split()
        if user == 'user1':
            alice.append(len(text))
        else:
            bob.append(len(text))
    if len(alice) == 0:
        alice.append(0)
    if len(bob) == 0:
        bob.append(0)
    return np.mean(alice), np.mean(bob)


def get_stop_words_share():
    stop_words = set(stopwords.words('english'))
    
    def stop_words_share(d):
        alice, bob = [], []
        for post in d['thread']:
            user, text = post['userId'], post['text'].lower().split()
            if user == 'user1':
                alice.extend(text)
            else:
                bob.extend(text)

        alice_stop = [w for w in alice if w in stop_words]
        bob_stop = [w for w in bob if w in stop_words]
        alice_share, bob_share = 0, 0
        if len(alice):
            alice_share = len(alice_stop) / float(len(alice))
        if len(bob):
            bob_share = len(bob_stop) / float(len(bob))
        return alice_share, bob_share
    
    return stop_words_share


# Fuzz QRatio
#
def min_QRatio(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    ratios = []
    for a in alice:
        for b in bob:
            ratios.append(fuzz.QRatio(a, b))
    if len(ratios) == 0:
        ratios.append(0)
    return min(ratios), min(ratios)


def max_QRatio(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    ratios = []
    for a in alice:
        for b in bob:
            ratios.append(fuzz.QRatio(a, b))
    if len(ratios) == 0:
        ratios.append(0)
    return max(ratios), max(ratios)


def avg_QRatio(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    ratios = []
    for a in alice:
        for b in bob:
            ratios.append(fuzz.QRatio(a, b))
    if len(ratios) == 0:
        ratios.append(0)
    return np.mean(ratios), np.mean(ratios)

def min_context_QRatio(d):
    alice, bob = [], []
    context = d['context']
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    aratios, bratios = [], []
    for a in alice:
        aratios.append(fuzz.QRatio(a, context))
    for b in bob:
        bratios.append(fuzz.QRatio(b, context))
    if len(aratios) == 0:
        aratios.append(0)
    if len(bratios) == 0:
        bratios.append(0)
    return min(aratios), min(bratios)


def min_WRatio(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    ratios = []
    for a in alice:
        for b in bob:
            ratios.append(fuzz.WRatio(a, b))
    if len(ratios) == 0:
        ratios.append(0)
    return min(ratios), min(ratios)


def max_WRatio(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    ratios = []
    for a in alice:
        for b in bob:
            ratios.append(fuzz.WRatio(a, b))
    if len(ratios) == 0:
        ratios.append(0)
    return max(ratios), max(ratios)


def avg_WRatio(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    ratios = []
    for a in alice:
        for b in bob:
            ratios.append(fuzz.WRatio(a, b))
    if len(ratios) == 0:
        ratios.append(0)
    return np.mean(ratios), np.mean(ratios)


def min_context_WRatio(d):
    alice, bob = [], []
    context = d['context']
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    aratios, bratios = [], []
    for a in alice:
        aratios.append(fuzz.WRatio(a, context))
    for b in bob:
        bratios.append(fuzz.WRatio(b, context))
    if len(aratios) == 0:
        aratios.append(0)
    if len(bratios) == 0:
        bratios.append(0)
    return min(aratios), min(bratios)


def get_human_words():
    hdict = set([w.strip() for w in open('dictionary.txt').readlines()])
    
    def human_words(d):
        alice, bob = [0], [0]
        for post in d['thread']:
            user, wc = post['userId'], len([w for w in post['text'].split() if w in hdict])
            if user == 'user1':
                alice.append(wc)
            else:
                bob.append(wc)
        return max(alice), max(bob)
    
    return human_words

def get_human_words_norm():
    hdict = set([w.strip() for w in open('dictionary.txt').readlines()])
    
    def human_words_norm(d):
        alice, bob = [0], [0]
        for post in d['thread']:
            user, wc = post['userId'], len([w for w in post['text'].split() if w in hdict]) / (1. + len(post['text'].split()))
            if user == 'user1':
                alice.append(wc)
            else:
                bob.append(wc)
        return max(alice), max(bob)
    
    return human_words_norm

def text2ngrams(text):
    return [''.join(ng) for ng in ngrams(text, 3, pad_left=True, left_pad_symbol=' ')]


def same_terms_count(l, r):
    return len(set(l).intersection(set(r)))


def ngram_intersection(d):
    context_ngrams = text2ngrams(d['context'])
    alice, bob = [0], [0]
    for post in d['thread']:
        text_ngrams = text2ngrams(post['text'])
        intersection = same_terms_count(context_ngrams, text_ngrams)
        if len(text_ngrams) == 0:
            return 0, 0
        int_share = float(intersection) / len(text_ngrams)
        (alice if post['userId'] == 'user1' else bob).append(int_share)
    return max(alice), max(bob)


def get_bot_words():
    bdict = set([w.strip() for w in open('antidictionary.txt').readlines()])
    
    def bot_words(d):
        alice, bob = [0], [0]
        for post in d['thread']:
            user, wc = post['userId'], len([w for w in post['text'].split() if w in bdict])
            if user == 'user1':
                alice.append(wc)
            else:
                bob.append(wc)
        return max(alice), max(bob)
    
    return bot_words

def get_bot_words_norm():
    bdict = set([w.strip() for w in open('antidictionary.txt').readlines()])
    
    def bot_words_norm(d):
        alice, bob = [0], [0]
        for post in d['thread']:
            user, wc = post['userId'], len([w for w in post['text'].split() if w in bdict]) / (1. + len(post['text'].split()))
            if user == 'user1':
                alice.append(wc)
            else:
                bob.append(wc)
        return max(alice), max(bob)
    
    return bot_words_norm

def min_partial_ratio(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    ratios = []
    for a in alice:
        for b in bob:
            ratios.append(fuzz.partial_ratio(a, b))
    if len(ratios) == 0:
        ratios.append(0)
    return np.min(ratios), np.min(ratios)

def min_context_partial_ratio(d):
    alice, bob = [], []
    context = d['context']
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    aratios, bratios = [], []
    for a in alice:
        aratios.append(fuzz.partial_ratio(a, context))
    for b in bob:
        bratios.append(fuzz.partial_ratio(b, context))
    if len(aratios) == 0:
        aratios.append(0)
    if len(bratios) == 0:
        bratios.append(0)
    return min(aratios), min(bratios)

def max_partial_ratio(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    ratios = []
    for a in alice:
        for b in bob:
            ratios.append(fuzz.partial_ratio(a, b))
    if len(ratios) == 0:
        ratios.append(0)
    return np.max(ratios), np.max(ratios)


def avg_partial_ratio(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    ratios = []
    for a in alice:
        for b in bob:
            ratios.append(fuzz.partial_ratio(a, b))
    if len(ratios) == 0:
        ratios.append(0)
    return np.mean(ratios), np.mean(ratios)


def dialogue_len(d):
    alice, bob = 0, 0
    for post in d['thread']:
        user = post['userId']
        if user == 'user1':
            alice += 1
        else:
            bob += 1
    return alice, bob

def dialog_len_share(d):
    alice, bob = dialogue_len(d)
    sum = float(alice+bob)
    if sum > 0:
        return alice/sum, bob/sum
    else:
        return 0, 0

def started_dialogue(d):
    alice, bob = 0, 0
    if len(d['thread']) > 0:
        if d['thread'][0]['userId'] == 'user1':
            alice = 1
        else:
            bob = 1
    return alice, bob

def finished_dialogue(d):
    alice, bob = 0, 0
    if len(d['thread']) > 0:
        if d['thread'][-1]['userId'] == 'user1':
            alice = 1
        else:
            bob = 1
    return alice, bob


def max_in_row(d):
    alice, bob = 0, 0
    max_alice, max_bob = 0, 0
    if len(d['thread']) > 0:
        for post in d['thread']:
            if post['userId'] == 'user1':
                max_bob = max(max_bob, bob); bob = 0
                alice += 1
            else:
                max_alice = max(max_alice, alice); alice=0
                bob += 1

    max_bob = max(max_bob, bob)
    max_alice = max(max_alice, alice)
    return max_alice, max_bob


def min_reply_time(d):

    def compute(arr):
        arr = np.asarray(arr)
        if arr.shape[0] < 2:
            return -1
        else:
            return np.min(arr[1:] - arr[:-1])

    alice, bob = [], []
    for post in d['thread']:
        try:
            user, timestamp = post['userId'], post['time']
        except:
            break
        if user == 'user1':
            alice.append(timestamp)
        else:
            bob.append(timestamp)
    return compute(alice), compute(bob)

def avg_reply_time(d):

    def compute(arr):
        arr = np.asarray(arr)
        if arr.shape[0] < 2:
            return -1
        else:
            return np.mean(arr[1:] - arr[:-1])

    alice, bob = [], []
    for post in d['thread']:
        try:
            user, timestamp = post['userId'], post['time']
        except:
            break
        if user == 'user1':
            alice.append(timestamp)
        else:
            bob.append(timestamp)
    return compute(alice), compute(bob)

def wmd_dist(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    res = wmd(' '.join(alice), ' '.join(bob))
    return res, res

def norm_wmd_dist(d):
    alice, bob = [], []
    for post in d['thread']:
        user, text = post['userId'], post['text']
        if user == 'user1':
            alice.append(text)
        else:
            bob.append(text)
    res = norm_wmd(' '.join(alice), ' '.join(bob))
    return res, res

def two_phrases_in_a_row(d):
    alice, bob = 0, 0
    for i, post in enumerate(d['thread'][1:]):
        if post['userId'] == d['thread'][i - 1]['userId']:
            if post['userId'] == 'user1':
                alice = 1
            else:
                bob = 1
    return alice, bob

def capital_letters_ratio(d):
    alice, bob = 0, 0
    for post in d['thread']:
        user, text = post['userId'], post['text']
        r = float(len(re.findall('[A-ZА-Я]+', text))) / (1. + len(text))
        if user == 'user1':
            alice = max(alice, r)
        else:
            bob = max(bob, r)
    return alice, bob

def question_marks(d):
    alice, bob = 0, 0
    for post in d['thread']:
        user, text = post['userId'], post['text']
        r = len(re.findall('\?', text))
        if user == 'user1':
            alice += r
        else:
            bob += r
    return alice, bob
 
def has_a_question(d):
    alice, bob = 0, 0
    for post in d['thread']:
        user, text = post['userId'], post['text'].lower()
        r = 1 if len(text) != 0 and (text[-1] == '?' or text.startswith(('what', 'when', 'where', 'who', 'why', 'which', 'are', 'do'))) else 0
        if user == 'user1':
            alice += r
        else:
            bob += r
    return alice, bob

def punctuation_marks(d):
    alice, bob = 0, 0
    for post in d['thread']:
        user, text = post['userId'], post['text']
        r = len(re.findall('[\.\,\!\?\-]', text))
        if user == 'user1':
            alice += r
        else:
             bob += r
    return alice, bob
