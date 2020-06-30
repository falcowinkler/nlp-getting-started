import nltk
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint


def transform_glove_embeddings(glove_file):
    glove2word2vec(glove_input_file=glove_file, word2vec_output_file="data/w2v_embeddings.txt")


# credit for loading the embeddings:
# https://github.com/ciwin/Intent_Classification/blob/master/Intent_classification_01.ipynb
def get_embeddings(filename):
    embeddings_index = {}
    dim = None
    with open(filename) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if dim is None:
                dim = len(coefs)
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index, dim


def create_embeddings_for_vocabulary(word_index, embeddings_index, embedding_dim):
    word_index = word_index.word_index
    # words not found in embedding index will be initialized randomly.
    embedding_matrix = np.random.uniform(-1, 1, (len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            print("Not in vacabulary: ", word)
    return embedding_matrix


def create_model(embeddings_matrix, vocab_size, embedding_dim, max_sent_length):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, weights=[embeddings_matrix], input_length=max_sent_length,
                        trainable=True))
    model.add(Bidirectional(LSTM(4)))  # best performance with a tiny encoding size!
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model
