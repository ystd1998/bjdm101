from nltk import ChunkParserI, ClassifierBasedTagger
from nltk.chunk import conlltags2tree, tree2conlltags
 
class FooChunkParser(ChunkParserI):
    def __init__(self, chunked_sents, **kwargs):
 
        # Transform the trees in IOB annotated sentences [(word, pos, chunk)]
        chunked_sents = [tree2conlltags(sent) for sent in chunked_sents]
 
        # Make tags compatible with the tagger interface [((word, pos), chunk)]
        def get_tagged_pairs(chunked_sent):
            return [((word, pos), chunk) for word, pos, chunk in chunked_sent]
        
        chunked_sents = [get_tagged_pairs(sent) for sent in chunked_sents]
 
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=chunked_sents,
            feature_detector=features,
            **kwargs)
 
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
        iob_triplets = [(word, token, chunk) for ((word, token), chunk) in chunks]
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)
 
chunker = FooChunkParser(train_sents)
print(chunker.evaluate(test_sents))