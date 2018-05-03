# https://www.commonlounge.com/discussion/926a4bbaae5b43ffad729742941fced1
from sklearn.feature_extraction.text import CountVectorizer
 
# list of text documents
text = ["this is test doc", "this is another test doc"]
 
# create the transform
vector = CountVectorizer()
 
# tokenize and build vocab
vector.fit(text)
 
# Print the summary
print(vector.vocabulary_)
 
# Transform document
X_Train = vector.transform(text)
 
# Print summary of transformed vector
print(X_Train.shape)
print(type(X_Train))
print(X_Train)
