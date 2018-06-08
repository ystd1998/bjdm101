# https://www.commonlounge.com/discussion/80d1d4e22c4f467f83445bef349d1ed2


grammar = nltk.CFG.fromstring("""S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'to' | 'my'
N -> 'market' | 'shorts'
V -> 'went'
P -> 'in'
""")
Here, S is the start symb