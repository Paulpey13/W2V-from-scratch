#Compare model and gensim results
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('model.txt', binary=False, encoding='ISO-8859-1')

# Résoudre une analogie
result = model.most_similar(positive=['roi', 'reine'], negative=['femme'], topn=1)
print(result)

