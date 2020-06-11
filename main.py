import download
import model
import tokenization
import dataset
import evaluation

MAX_SENTENCE_LENGTH = 20
BATCH_SIZE = 32
MODEL_PATH = 'saved_models/nlp_classifier_word2vec.h5'
PRETRAINED_EMBEDDINGS = "glove.6B.100d"

dataset_path = download.download_from_kaggle()
EMBEDDINGS_FILE = download.download_word_vectors(f"{PRETRAINED_EMBEDDINGS}.txt")
df = dataset.load_dataset(dataset_path)

tokenizer = tokenization.create_tokenizer(df)
features, labels = tokenization.transform_dataset(df, tokenizer, MAX_SENTENCE_LENGTH)
embeddings_index, embeddings_dim = model.get_embeddings(EMBEDDINGS_FILE)
embeddings_matrix = model.create_embeddings_for_vocabulary(tokenizer, embeddings_index, embeddings_dim)

vocab_size = len(embeddings_matrix)

nn_model = model.create_model(embeddings_matrix, vocab_size, embeddings_dim, MAX_SENTENCE_LENGTH)

encoded_words_train, encoded_words_test, labels_train, labels_test = dataset.train_test_split(features, labels)

nn_model.fit(encoded_words_train, labels_train, epochs=20, batch_size=BATCH_SIZE)
nn_model.save(MODEL_PATH)

print('Done training!')

test_loss, test_acc = evaluation.evaluate(MODEL_PATH, encoded_words_test, labels_test)

print("Loss: ", test_loss, "Accuracy: ", test_acc)
