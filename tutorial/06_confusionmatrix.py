# https://www.commonlounge.com/discussion/926a4bbaae5b43ffad729742941fced1
from sklearn.metrics import confusion_matrix
expected = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
results = confusion_matrix(expected, predicted)
print(results)

from sklearn.metrics import classification_report
report = classification_report(expected, predicted)
print(report)
