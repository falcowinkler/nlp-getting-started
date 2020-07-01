Binary tweet classification

This was an experiment to use word vectors for a simple classification task.
Idea:

- Map every word to it's corresponding word vector (we use globe embeddings)
- Feed the embeddings into a bidirectional LSTM
- Compute output by concatenating the architecture with a dense layer

Validation score on kaggle is around .7 ~ .78

Credits for the hyperparameters go to https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

Credits for the code to load glove embeddings: https://github.com/ciwin/Intent_Classification

### Run

```bash
export KAGGLE_USERNAME=<your-kaggle-username>
export KAGGLE_KEY=<your-kaggle-api-key>
python main.py
```