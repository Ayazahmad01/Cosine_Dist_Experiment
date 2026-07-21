# Cosine Distance vs Other Distance Metrics for Text Document Similarity

**Author:** Ayaz Ahmad

This project compares four distance measures — **Cosine, Euclidean, Manhattan, and Jaccard** — for finding similar text documents. Documents are turned into TF-IDF vectors and classified with a k-Nearest Neighbors classifier. The best measure is the one that classifies documents most correctly.

## Files

| File | What it is |
|------|------------|
| `cosineDis.py` | The experiment code |
| `Dataset.csv` | Sample dataset (60 documents, 4 categories) |
| `cosine_distance_paper.docx` | The research paper |

## How to run

1. Install the libraries (one time):
   ```
   pip install scikit-learn pandas matplotlib
   ```
2. Keep `cosineDis.py` and `Dataset.csv` in the same folder, then run:
   ```
   python cosineDis.py
   ```
3. It prints a results table and saves:
   - `results_table.csv` — the scores
   - `f1_comparison.png` — a bar chart

## Use the bigger dataset (optional)

For stronger results, open `cosineDis.py` and change:
```python
USE_20NEWSGROUPS = False
```
to
```python
USE_20NEWSGROUPS = True
```
This uses the 20 Newsgroups dataset (thousands of documents). It downloads automatically the first time and needs an internet connection.

## Result

On the sample dataset, **cosine distance** performed best, because it compares the direction of document vectors rather than their length. This makes it well suited to text, where documents vary in length.

## License

MIT
