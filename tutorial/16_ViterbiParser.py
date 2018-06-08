
# https://www.commonlounge.com/discussion/80d1d4e22c4f467f83445bef349d1ed2

from __future__ import print_function


import nltk
from nltk.grammar import Nonterminal
from nltk.corpus import treebank
# get training data
training_set = treebank.parsed_sents()
 
# example training sentence
print(training_set[1])

# Extract the rules for all annotated training sentences
rules = list(set(rule for sent in training_set for rule in sent.productions()))
print(rules[0:5])
# [VBZ -> 'cites', VBD -> 'spurned', PRN -> , ADVP-TMP ,, NNP -> 'ACCOUNT', JJ -> '36-day']

sentence = "The famous algorithm produced accurate results"
# get tokens and their POS tags
from pattern.en import tag as pos_tagger
tagged_sent = pos_tagger(sentence)
for word, tag in tagged_sent:
    t = nltk.Tree.fromstring("("+ tag + " " + word +")")
    for rule in t.productions():
        rules.append(rule)
        
# build the parser
viterbi_parser = nltk.ViterbiParser(treebank_grammar)
# get sample sentence tokens
tokens = nltk.word_tokenize(sentence)
# get parse tree
result = list(viterbi_parser.parse(tokens))
print(result[0])

        