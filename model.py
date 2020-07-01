import numpy as np
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.models import Sequential
from keras.optimizers import Adam


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
    # words not found in embedding index need to be zero vectors in this case.
    # random vectors for unknows greatly confused the model it seems
    # embedding_matrix = np.random.uniform(-1, 1, (len(word_index) + 1, embedding_dim))
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim), dtype=np.float32)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_model(embeddings_matrix, vocab_size, embedding_dim, max_sent_length):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, weights=[embeddings_matrix], input_length=max_sent_length,
                        trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(
        LSTM(64, dropout=0.2, recurrent_dropout=0.2
             ))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-5), metrics=["accuracy"])
    model.summary()
    return model
