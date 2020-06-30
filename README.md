Binary tweet classification

This was an experiment to use word vectors for a simple classification task.
Idea:

- Map every word to it's corresponding word vector (we use globe embeddings)
- Feed the embeddings into a bidirectional LSTM
- Compute output by concatenating the architecture with a dense layer

Performance is about the same as logistic regression.. for now

### Run

```bash
export KAGGLE_USERNAME=<your-kaggle-username>
export KAGGLE_KEY=<your-kaggle-api-key>
python main.py
```