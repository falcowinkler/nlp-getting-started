import re

import nltk
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')
text_canonize_spec = [
    {
        "pattern": r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
        "placeholder": "internet"
    },
    {
        "pattern": r"@[A-Za-z0-9]+\b",
        "placeholder": "person"
    },
    {
        "pattern": r"[0-9]+[0-9,.]*\b",
        "placeholder": "number"
    },
    {
        "pattern": "[^\\sa-zA-Z]+",
        "placeholder": ""
    }
]


def text_canonize(canonical_text):
    for spec in text_canonize_spec:
        canonical_text = re.sub(spec["pattern"], spec["placeholder"], canonical_text)
    return canonical_text.lower()


def cleanse(line):
    # TODO can we use named entity recognition for location, as an indicator / feature?
    """Takes a tweet and turns it into whitespace separated string of tokens"""
    line = text_canonize(line)
    words = line.split()
    words = [word.lower() for word in words if word not in set(stopwords.words('english'))]
    line = ' '.join(words)
    return line


def create_tokenizer(df):
    tokenizer = Tokenizer(split=' ',
                          # num_words=3000
                          )
    tokenizer.fit_on_texts(df.text)
    return tokenizer


def transform_dataset(df, max_sent_length):
    df.text = df.text.map(cleanse)
    tokenizer = create_tokenizer(df)
    encoded_tweets = tokenizer.texts_to_sequences(df.text)
    padded_sentences = pad_sequences(encoded_tweets, maxlen=max_sent_length, padding="post")
    labels = np.array(df.target) if 'target' in df else None
    return padded_sentences, labels
