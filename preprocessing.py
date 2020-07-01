# Source: https://www.kaggle.com/life2short/data-processing-replace-abbreviation-of-word
import re
import string

import nltk
import pandas as pd

import dataset
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
punctuation = r'["\'?,\.\!-:;]+'
from nltk.tokenize import word_tokenize

abbr_dict = {
    "what's": "what is",
    "what're": "what are",
    "who's": "who is",
    "who're": "who are",
    "where's": "where is",
    "where're": "where are",
    "when's": "when is",
    "when're": "when are",
    "how's": "how is",
    "how're": "how are",

    "i'm": "i am",
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "there're": "there are",

    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
    "they've": "they have",
    "who've": "who have",
    "would've": "would have",
    "not've": "not have",
    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "they'll": "they will",

    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "can't": "can not",
    "couldn't": "could not",
    "don't": "do not",
    "didn't": "did not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",

    # Twitter specific stuff
    # 'https?://\S+|www\.\S+': '',
    r'https?:\/\/(?:www\.)?t\.co\/([a-zA-Z0-9_]+)\b': 'tweet',
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)': '',
    '<.*?>': 'html',
    r"@[A-Za-z0-9]+\b": "mention",
    "[0-9]+[0-9,.]*": "number",
    punctuation: ' ',
    r'\s+': ' ',  # replace multi space with one single space
    'mh370': 'malaysa airlines flight',
    'yrs': 'years'
}

from autocorrect import Speller

spell = Speller(lang='en', fast=True)
stemmer = nltk.SnowballStemmer('english')
lemmatizer = nltk.WordNetLemmatizer()

stopwords = set(stopwords.words('english'))


def clean_text(df):
    df.text = df.text.map(spell)
    df.text = df.text.str.lower()
    df.replace(abbr_dict, regex=True, inplace=True)
    df.text = df.text.map(cleanse)
    return df


def cleanse(line):
    # TODO can we use named entity recognition for location, as an indicator / feature?
    """Takes a tweet and turns it into whitespace separated string of tokens"""
    words = [
        word.lower() for word in word_tokenize(line)
        if (word not in stopwords) and word.isalpha()]
    line = ' '.join(words)
    return line


if __name__ == '__main__':
    # Playground for preprocessing
    df = dataset.load_dataset('data/test.csv')
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_colwidth', 1000)
    df = df.filter(['text'])
    print(df.head(500))
    clean_text(df)
    print(df.head(500))
