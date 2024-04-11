from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

# glove_input_file = '..\\word2vec_model\\glove.42B.300d.txt'
# word2vec_output_file = '..\\word2vec_model\\glove.42B.300d.word2vec.txt'
glove_input_file = '..\\word2vec_model\\glove.6B\\glove.6B.300d.txt'
word2vec_output_file = '..\\word2vec_model\\glove.6B.300d.word2vec.txt'

glove2word2vec(glove_input_file, word2vec_output_file)
