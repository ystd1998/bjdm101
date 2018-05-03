sentence='Pattern library can extract good chunks from a sentence'
from pattern.en import parsetree
tree=parsetree(sentence)
# print the chunks from shallow parsed sentence tree
for node in tree:
    for chunk in node.chunks:
        print chunk.type, [(word.string, word.type) for word in chunk.words]
