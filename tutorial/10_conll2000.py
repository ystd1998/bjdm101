from nltk.corpus import conll2000
import random
 
conll_data = list(conll2000.chunked_sents())
random.shuffle(conll_data)
train_sents = conll_data[:int(len(conll_data) * 0.8)]
test_sents = conll_data[int(len(conll_data) * 0.8 + 1):]