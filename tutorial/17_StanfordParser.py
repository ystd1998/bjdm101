
# https://www.commonlounge.com/discussion/80d1d4e22c4f467f83445bef349d1ed2

from nltk.parse.stanford import StanfordParser
# create parser object
scp = StanfordParser(path_to_jar='/path/to/stanford-parser.jar', path_to_models_jar='path/to/stanford-parser-models.jar')
# get parse tree
result = list(scp.raw_parse(sentence))

