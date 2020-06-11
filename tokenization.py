import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def cleanse(line):
    # TODO can we use named entity recognition for location, as an indicator / feature?
    """Takes a tweet and turns it into whitespace seperated string of tokens"""
    ps = PorterStemmer()
    line = re.sub("[^a-zA-Z]", ' ', line)
    words = line.split()
    words = [word for word in words if not word in set(stopwords.words('english'))]
    words = [ps.stem(word) for word in words]
    line = ' '.join(words)
    return line


def create_tokenizer(df):
    tokenizer = Tokenizer(split=' ')
    tokenizer.fit_on_texts(df.text)
    return tokenizer


def transform_dataset(df, tokenizer, max_sent_length):
    encoded_tweets = tokenizer.texts_to_sequences(df.text)
    padded_sentences = pad_sequences(encoded_tweets, maxlen=max_sent_length, padding="post")
    labels = np.array(df.target)
    return padded_sentences, labels
