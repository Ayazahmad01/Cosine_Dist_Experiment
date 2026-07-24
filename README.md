# Performance Evaluation of Cosine Distance and Traditional Distance Metrics for TF-IDF-Based Text Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21531416.svg)](https://doi.org/10.5281/zenodo.21531416)

This repository contains the source code, datasets, and experimental results for the research paper:

**Performance Evaluation of Cosine Distance and Traditional Distance Metrics for TF-IDF-Based Text Classification**

📄 **Preprint (Zenodo):** https://doi.org/10.5281/zenodo.21531416

---

## Overview

This study evaluates the performance of four distance metrics for text classification using **TF-IDF** features and the **k-Nearest Neighbors (k-NN)** algorithm.

The following distance measures are compared:

* Cosine Distance
* Euclidean Distance
* Manhattan Distance
* Jaccard Distance

The experiments are conducted on the public **20 Newsgroups** dataset using **Scikit-learn**.

---

## Dataset

The dataset is downloaded automatically from **Scikit-learn** during the first execution.

Selected categories:

* `comp.graphics`
* `rec.sport.baseball`
* `sci.med`
* `talk.politics.guns`

Dataset Statistics:

* Total Documents: **2,256**
* Training Samples: **1,579**
* Testing Samples: **677**

Headers, footers, and quoted replies are removed to ensure classification is based only on document content.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Ayazahmad01/Cosine_Dist_Experiment.git

cd Cosine_Dist_Experiment
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python twenty_newsgroups_experiment.py
```

---

## Output

The experiment generates:

* `results_table.csv`
* `f1_comparison.png`

---

## Experimental Results

| Distance Metric | Accuracy | Precision | Recall | F1-Score |
| --------------- | -------: | --------: | -----: | -------: |
| Cosine          |    0.900 |     0.905 |  0.900 |    0.899 |
| Jaccard         |    0.808 |     0.822 |  0.808 |    0.808 |
| Euclidean       |    0.479 |     0.548 |  0.479 |    0.362 |
| Manhattan       |    0.266 |     0.137 |  0.266 |    0.156 |

**Conclusion:** Cosine Distance achieved the highest classification performance, making it the most effective distance metric for TF-IDF-based text classification in this study.

---

## Reproducibility

* Python 3.x
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

Jaccard Distance is computed on binarized TF-IDF features because it is defined for set-based representations.

Results may vary slightly depending on the Scikit-learn version.

---

## Citation

If you use this repository in your research, please cite:

> Ayaz Ahmad. *Performance Evaluation of Cosine Distance and Traditional Distance Metrics for TF-IDF-Based Text Classification*. Zenodo. https://doi.org/10.5281/zenodo.21531416

---

## Author

**Ayaz Ahmad**

BS Computer Science

University of Engineering and Technology (UET) Peshawar, Pakistan

📧 Email: [24pwbcs1197@uetpeshawar.edu.pk](mailto:24pwbcs1197@uetpeshawar.edu.pk)

**ORCID:** https://orcid.org/0009-0000-8651-2485

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
