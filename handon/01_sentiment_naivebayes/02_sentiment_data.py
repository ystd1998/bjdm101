# https://www.commonlounge.com/discussion/604fb452d5b44a9688267823ac1b4d0a

import cPickle
import numpy as np

f = open('sentiment_data.pkl', 'rb')
train_positive, train_negative, test_positive, test_negative = cPickle.load(f)
f.close()

print('Data description ... ')
print(len(train_positive), len(train_negative), len(test_positive), len(test_negative))
print('='*120)
print(train_positive[:10])
print('='*120)
print(train_negative[:10])
print('='*120)

import re
import string

from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

###############################################################################
## helper functions

table = string.maketrans('', '')

def review_tokens(review):
#    return [token.lower() for token in re.findall('[A-Za-z]+', review)]  
    # tokens = word_tokenize(review)

    # Convert words to lower case
    # tokens = [w.lower() for w in tokens]
    tokens = [token.lower() for token in re.findall('[A-Za-z]+', review)]

    # Remove punctuation from each word
    # stripped = [w.translate(table, string.punctuation) for w in tokens]

    # Filter out stop words
    return [w for w in tokens if not w in stop_words]


###############################################################################
## naive bayes

nneg, npos = len(train_negative), len(train_positive)
print (nneg, npos)

## train
# P(positive) and P(negative)
pos_prob = float(npos)/(npos+nneg)
neg_prob = float(nneg)/(npos+nneg)


# P(positive|token) and P(negative|token)
from collections import Counter
from math import log

counts_pos = Counter()
counts_neg = Counter()
for r in train_positive:
    ts = review_tokens(r)
    counts_pos.update( ts )
for r in train_negative:
    ts = review_tokens(r)
    counts_neg.update( ts )

# predict
all_test = np.concatenate((test_positive, test_negative))
labels = [True]*len(test_positive) + [False]*len(test_negative)
 
predictions = list()
for review in all_test:
    pos = log(pos_prob)
    neg = log(neg_prob)
    for token in review_tokens(review):
        # update pos, neg 
        ttl = counts_pos[token]+counts_neg[token]+2
        if ttl > 0:
            pos += log(float(counts_pos[token]+1)/ttl)
            neg += log(float(counts_neg[token]+1)/ttl)
    predictions.append(pos > neg)
 
# evaluate
print(classification_report(labels, predictions))
print('='*120)
print(confusion_matrix(labels, predictions))
print('='*120)

vocab = set([token for review in np.concatenate((train_positive, train_negative)) for token in review_tokens(review)])
positivity = dict()
for token in vocab:
    ttl = counts_pos[token]+counts_neg[token]+2
    positivity[token] = (float(counts_pos[token]+1)/ttl)

print('Most positive tokens', sorted(positivity.keys(), key=positivity.get, reverse=True)[:10])
print('Most negative tokens', sorted(positivity.keys(), key=positivity.get, reverse=False)[:10])
