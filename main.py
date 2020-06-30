import nltk

import download
import model
import tokenization
import dataset
import evaluation
import numpy as np
import pandas as pd
import json

with open('config.json') as config_file:
    config = json.load(config_file)

MAX_SENTENCE_LENGTH = config['max_sentence_length']
BATCH_SIZE = config['batch_size']
MODEL_PATH = config['model_path']
PRETRAINED_EMBEDDINGS = config['pretrained_embeddings']
EPISODES = config['episodes']


def word_index_for_all_data(train_dataset_path, test_dataset_path):
    """create workd index for train and test (kaggle submission) dataset
    Otherwise the embedding matrix causes problems when re-loading the model.
    """
    _, _, _, df1 = create_dataset(train_dataset_path)
    _, _, _, df2 = create_dataset(test_dataset_path)
    df1 = df1.filter(['text'])
    df2 = df2.filter(['text'])
    df1.text = df1.text.map(tokenization.cleanse)
    df2.text = df2.text.map(tokenization.cleanse)
    return tokenization.create_tokenizer(pd.concat([df1, df2]))


def create_dataset(dataset_path):
    df = dataset.load_dataset(dataset_path)
    tokenizer = tokenization.create_tokenizer(df)
    features, labels = tokenization.transform_dataset(df, MAX_SENTENCE_LENGTH)
    return features, labels, tokenizer, df


def create_model(word_index):
    embeddings_dir = download.download_word_vectors()
    embeddings_file = f"{embeddings_dir}/{PRETRAINED_EMBEDDINGS}.txt"
    embeddings_index, embeddings_dim = model.get_embeddings(embeddings_file)
    embeddings_matrix = model.create_embeddings_for_vocabulary(word_index, embeddings_index, embeddings_dim)
    vocab_size = len(embeddings_matrix)
    return model.create_model(embeddings_matrix, vocab_size, embeddings_dim, MAX_SENTENCE_LENGTH)


def train(word_index):
    dataset_path = download.download_from_kaggle()
    features, labels, tokenizer, _ = create_dataset(dataset_path)
    encoded_words_train, encoded_words_test, labels_train, labels_test = dataset.train_test_split(features, labels)
    nn_model = create_model(word_index)
    nn_model.fit(encoded_words_train, labels_train, epochs=EPISODES, batch_size=BATCH_SIZE)
    nn_model.save(MODEL_PATH)

    print('Done training!')

    test_loss, test_acc = evaluation.evaluate(MODEL_PATH, encoded_words_test, labels_test)

    print("Loss: ", test_loss, "Accuracy: ", test_acc)


def predict(test_dataset_path, output_path, model_path, word_index):
    test_features, _, tokenizer, df = create_dataset(test_dataset_path)
    nn_model = create_model(word_index)
    nn_model.load_weights(model_path)
    predictions = list(map(int, np.around(nn_model.predict(test_features).squeeze())))
    output_dataset = df.filter(['id'])
    output_dataset['target'] = predictions
    output_dataset.to_csv(output_path, sep=",", index=False)
    for feature, prediction in zip(df.text, predictions):
        print(feature, prediction)


if __name__ == '__main__':
    word_index = word_index_for_all_data('data/train.csv', 'data/test.csv')
    # predict('data/test.csv', 'data/output.csv', 'saved_models/nlp_classifier_word2vec.h5', word_index)
    train(word_index)
