import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import dataset
import preprocessing


def create_tokenizer(df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.text)
    return tokenizer


def transform_dataset(df, max_sent_length, tokenizer):
    df = preprocessing.clean_text(df)
    encoded_tweets = tokenizer.texts_to_sequences(df.text)
    padded_sentences = pad_sequences(encoded_tweets, maxlen=max_sent_length, padding="post", truncating='post')
    labels = np.array(df.target) if 'target' in df else None
    return padded_sentences, labels
