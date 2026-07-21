USE_20NEWSGROUPS = True    
SAMPLE_CSV = "Dataset.csv"      
K_NEIGHBORS = 5                 
TEST_SIZE = 0.30                
RANDOM_STATE = 42              

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt

def load_data():
    if USE_20NEWSGROUPS:
        print("Loading the 20 Newsgroups dataset (downloads on first run)...")
        from sklearn.datasets import fetch_20newsgroups
        categories = [
            "rec.sport.baseball",   # sports
            "comp.graphics",        # technology
            "sci.med",              # health
            "talk.politics.guns",   # politics
        ]
        train = fetch_20newsgroups(subset="train", categories=categories,
                                   remove=("headers", "footers", "quotes"),
                                   random_state=RANDOM_STATE)
        texts = train.data
        labels = [train.target_names[t] for t in train.target]
        df = pd.DataFrame({"text": texts, "category": labels})
        df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    else:
        print(f"Loading the dataset from '{SAMPLE_CSV}'...")
        df = pd.read_csv(SAMPLE_CSV)

    print(f"  Loaded {len(df)} documents across {df['category'].nunique()} categories.")
    print(f"  Categories: {sorted(df['category'].unique())}")
    return df["text"].tolist(), df["category"].tolist()
def vectorize(train_texts, test_texts):
    """
    TfidfVectorizer cleans the text AND does the TF-IDF math:
      - lowercase the text
      - remove common English stop words (the, and, is, ...)
      - build TF-IDF number vectors
    We 'fit' only on training text so the test set stays unseen (fair).
    """
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", min_df=1)
    X_train = vectorizer.fit_transform(train_texts)   
    X_test = vectorizer.transform(test_texts)         
    return X_train, X_test

def jaccard_distance(a, b):
    """
    Jaccard looks at WHICH words appear, ignoring how often.
    Any value > 0 counts as 'word present'.
    Jaccard distance = 1 - (shared words / total unique words).
    """
    a_present = a > 0
    b_present = b > 0
    intersection = np.logical_and(a_present, b_present).sum()
    union = np.logical_or(a_present, b_present).sum()
    if union == 0:
        return 1.0
    return 1.0 - (intersection / union)
def evaluate_metric(metric_name, X_train, X_test, y_train, y_test):
    """Train k-NN with the given distance, predict, and compute four scores."""
    if metric_name == "jaccard":
        knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, metric=jaccard_distance)
        Xtr, Xte = X_train.toarray(), X_test.toarray() 
    else:
        knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, metric=metric_name)
        Xtr, Xte = X_train, X_test

    knn.fit(Xtr, y_train)
    y_pred = knn.predict(Xte)

    return {
        "Metric": metric_name.capitalize(),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

def main():
    print("=" * 70)
    print("Cosine Distance vs Other Metrics  —  by Ayaz Ahmad")
    print("=" * 70)

    texts, labels = load_data()
    train_texts, test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=labels)
    print(f"  Training documents: {len(train_texts)}  |  Testing documents: {len(test_texts)}")

    # Convert category names to numbers 
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    X_train, X_test = vectorize(train_texts, test_texts)
    print(f"  Vocabulary size (features): {X_train.shape[1]}")
    print("-" * 70)

    # Evaluate each distance measure
    metrics = ["cosine", "euclidean", "manhattan", "jaccard"]
    results = []
    for m in metrics:
        print(f"Running k-NN with '{m}' distance...")
        results.append(evaluate_metric(m, X_train, X_test, y_train, y_test))

    df = pd.DataFrame(results).set_index("Metric").round(4)
    print("\n" + "=" * 70)
    print("RESULTS  (put these numbers in Table 1 of your paper)")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)
    print(f"Best distance measure by F1-Score: {df['F1-Score'].idxmax()}")
    df.to_csv("results_table.csv")
    print("Saved: results_table.csv")
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(df.index, df["F1-Score"],
                   color=["#2E86C1", "#5DADE2", "#85C1E9", "#AED6F1"])
    plt.ylabel("F1-Score")
    plt.title("F1-Score by Distance Measure (Text Document Similarity)")
    plt.ylim(0, 1)
    for bar, val in zip(bars, df["F1-Score"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("f1_comparison.png", dpi=150)
    print("Saved: f1_comparison.png  (paste this chart into your paper)")
    print("\nDone! Now put these real numbers in Table 1 of your Word paper.")

if __name__ == "__main__":
    main()
