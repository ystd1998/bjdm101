# https://www.commonlounge.com/discussion/97d603f60c044bcd8fc98991e515e29d

from nltk.stem.porter import PorterStemmer
 
def features(tokens, index, history):
    # tokens are tagged words of a sentence
    # Index is the index of token for which the features to be extracted
    # history is the previous predicted tags
    
    stemmer = PorterStemmer()
 
    # Build the sequence of words for training
    tokens = [('__PREVSEQ2__', '__PREVSEQ2__'), 
        ('__PREVSEQ1__', '__PRESEQ1__')] + list(tokens) + [('__END1__', '__END1__'), 
        ('__END2__', '__END2__')]
    history = ['__PREVSEQ2__', '__PREVSEQ2__'] + list(history)
 
    # shift the index with 2 to point to current token
    index += 2
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prev2word, prev2pos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    next2word, next2pos = tokens[index + 2]
 
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
 
        'next-word': nextword,
        'next-pos': nextpos,
 
        'next-next-word': next2word,
        'next-next-pos': next2pos,

                'prev-word': prevword,
        'prev-pos': prevpos,
 
        'prev-prev-word': prev2word,
        'prev-prev-pos': prev2pos,
    }