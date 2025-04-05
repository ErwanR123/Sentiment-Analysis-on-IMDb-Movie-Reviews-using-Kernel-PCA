# Sentiment Analysis on IMDb Movie Reviews using Kernel PCA

This repository presents a complete machine learning pipeline for sentiment analysis, with a particular focus on evaluating the impact of non-linear dimensionality reduction via Kernel PCA (cosine kernel) on model performance.
This work is part of the experimental section of a Master's thesis focused on Kernel PCA, conducted in the context of a Master's degree in Applied Mathematics at Université Paris-Dauphine.
## Project Goal

The primary objective of this project is not only to classify IMDb reviews as positive or negative, but more importantly to determine whether applying Kernel PCA improves model performance on high-dimensional textual data.

To this end, multiple models are trained and compared:
- with raw Bag-of-Words vectors (5000 dimensions),
- and with projections in a lower-dimensional space (2300 dimensions) via Kernel PCA using a cosine kernel.

## Pipeline Overview

### 1. Text Preprocessing
- HTML tag removal, punctuation cleaning, digit filtering
- Lowercasing
- Tokenization with POS tagging
- Lemmatization (WordNet via NLTK)
- Stopword filtering

### 2. Vectorization
- `CountVectorizer(max_features=5000)`
- Bag-of-Words representation: $\mathbf{x} \in \mathbb{N}^{5000}$

### 3. Dimensionality Reduction
- `KernelPCA(n_components=2300, kernel='cosine')`
- Number of components selected by minimizing reconstruction error

### 4. Classification
Models tested (with and without Kernel PCA):
- Logistic Regression
- SVM (RBF kernel)
- K-Nearest Neighbors (KNN)
- Also tested: Random Forest, MLP, Gradient Boosting

## Performance Comparison

| Model               | KPCA? | Accuracy | F1-score | AUC   |
|---------------------|--------|----------|----------|-------|
| Logistic Regression | No     | 0.84     | 0.84     | 0.91  |
| Logistic Regression | Yes    | 0.86     | 0.85     | 0.93  |
| SVM (RBF)           | Yes    | 0.86     | 0.85     | 0.93  |
| KNN                 | No     | 0.60     | 0.59     | 0.68  |
| KNN                 | Yes    | 0.69     | 0.68     | 0.73  |

Kernel PCA improves the performance of KNN significantly, and slightly boosts logistic regression and SVM as well — confirming that non-linear projections can enhance class separability for certain models.

## Try Your Own Review

You can test the model in interactive mode with your own input using the `predict_comment()` function defined in the script.

Example:
Enter a review: "The plot was surprisingly moving and the lead actor delivered a powerful performance."

Result: POSITIVE (93.25% confidence)


## Project Files

| File | Description |
|------|-------------|
| `sentiment_analysis.py` | Full pipeline (preprocessing → vectorization → KPCA → models) |
| `nltk_data/` | NLTK lemmatizer, stopwords, POS-tagger resources |
| `IMDB Dataset.csv` | Dataset used for training/testing |
| `README.md` | Project documentation |
| `LICENSE` | MIT License |

## References

- [IMDb Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Kernel PCA (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)

## Author

Developed by **Erwan Ouabdesselam** as part of my Master's thesis in Applied Mathematics, Université Paris-Dauphine, 2025.

## License

This project is licensed under the [MIT License](LICENSE).
