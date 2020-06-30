import os
import zipfile

import kaggle

kaggle.api.authenticate()


def download_from_kaggle():
    dataset_path = 'data/nlp-getting-started.zip'
    if not os.path.exists(dataset_path):
        kaggle.api.competition_download_files('nlp-getting-started', path="data")
    with zipfile.ZipFile('data/nlp-getting-started.zip', 'r') as zipObj:
        zipObj.extractall(path='data')
    return 'data/train.csv'


def download_word_vectors():
    target_path = "data/word_vectors"
    if not os.path.exists(target_path):
        kaggle.api.dataset_download_files("rtatman/glove-global-vectors-for-word-representation",
                                          path=target_path,
                                          unzip=True)
    return target_path
