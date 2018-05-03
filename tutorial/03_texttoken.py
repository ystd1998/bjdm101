# Split text into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
 
# Convert words to lower case
tokens = [w.lower() for w in tokens]
 
# Remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
 
# Remove tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
 
# Filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
 
# Stemming of words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
