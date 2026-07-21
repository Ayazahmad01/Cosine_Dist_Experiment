# Cosine Distance vs. Traditional Distance Metrics for TF-IDF Text Classification

Code and results for the paper *"Performance Evaluation of Cosine Distance and
Traditional Distance Metrics for TF-IDF-Based Text Classification."*

The experiment compares four distance measures — **cosine, Euclidean, Manhattan,
and Jaccard** — for classifying text with a k-Nearest Neighbors classifier over
TF-IDF features, using the public **20 Newsgroups** benchmark.

## Dataset

20 Newsgroups is fetched automatically by scikit-learn on first run, so **no data
files need to be downloaded manually**. The experiment uses four categories:

- `comp.graphics` (technology)
- `rec.sport.baseball` (sports)
- `sci.med` (health)
- `talk.politics.guns` (politics)

This gives 2,256 documents. The dataset's predefined split is used: 1,579 for
training and 677 for testing. Headers, footers, and quoted reply text are removed
so the classifier learns from the actual content rather than metadata.

## How to run

```bash
pip install -r requirements.txt
python twenty_newsgroups_experiment.py
```

The first run downloads the dataset (~14 MB) and caches it locally. The script
prints a results table and saves `results_table.csv` and `f1_comparison.png`.

## Results

k-NN with k = 5, macro-averaged metrics on the 677-document test set:

| Measure    | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| Cosine     | 0.900    | 0.905     | 0.900  | 0.899    |
| Jaccard    | 0.808    | 0.822     | 0.808  | 0.808    |
| Euclidean  | 0.479    | 0.548     | 0.479  | 0.362    |
| Manhattan  | 0.266    | 0.137     | 0.266  | 0.156    |

Cosine distance performs best because it compares vector direction rather than
magnitude, making it robust to differences in document length. Jaccard runs on the
binarized (present/absent) form of the features, since it is a set measure.

## Requirements

Python 3, plus the packages listed in `requirements.txt`.

## Notes on reproducibility

Jaccard distance is computed on binarized features (term present/absent) because it
is defined on sets, while the other three measures use the TF-IDF vectors directly.
Exact scores may shift slightly with different scikit-learn versions.
