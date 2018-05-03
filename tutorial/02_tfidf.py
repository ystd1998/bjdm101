from sklearn.feature_extraction.text import TfidfVectorizer
import operator
 
corpus=["this car got the excellence award",\
         "good car gives good mileage",\
         "this car is very expensive",\
         "the company is growing with very high production",\
         "this company is financially good"]
 
vocabulary = set()
for doc in corpus:
    vocabulary.update(doc.split())
 
vocabulary = list(vocabulary)
word_index = {w: idx for idx, w in enumerate(vocabulary)}
 
tfidf = TfidfVectorizer(vocabulary=vocabulary)
 
# Fit the TfIdf model
tfidf.fit(corpus)
tfidf.transform(corpus)
 
for doc in corpus:
    score={}
    print doc
    # Transform a document into TfIdf coordinates
    X = tfidf.transform([doc])
    for word in doc.split():
        score[word] = X[0, tfidf.vocabulary_[word]]
    sortedscore = sorted(score.items(), key=operator.itemgetter(1), reverse=True)
    print "\t", sortedscore
