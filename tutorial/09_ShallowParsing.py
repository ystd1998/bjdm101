import nltk
sentence = "The famous algorithm produced accurate results"
tokens = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokens, tagset='universal')
cp = nltk.RegexpParser(grammar)
result = cp.parse(tagged_sent)
print(result)
