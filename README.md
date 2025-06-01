# Sentiment Analysis on IMDb Movie Reviews using Kernel PCA

This project investigates the impact of **nonlinear dimensionality reduction** via **Kernel PCA** on the performance of classic classification models in a sentiment analysis task using IMDb movie reviews.

---

## Project Overview

The pipeline includes:

- Text preprocessing: tokenization, lemmatization, POS-tagging, stopword removal (using NLTK)
- Vectorization using **Bag-of-Words** (`CountVectorizer`) with restricted vocabulary
- Dimensionality reduction using **Kernel PCA** with a cosine kernel
- Training and evaluation of multiple classifiers: K-Nearest Neighbors, Logistic Regression, SVM
- Sensitivity analysis on the number of components
- Comparison with models trained without dimensionality reduction and with classical PCA

---

## Full Report

This work is part of a broader Master's thesis on Kernel PCA. The full methodology and experimental results are available in the PDF report (in french):

[Master_Thesis.pdf](./Master_Thesis.pdf)  
The IMDb sentiment classification experiment is detailed in **Section 4** of the document.

---

## References

- [IMDb Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Kernel PCA (Scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)

---

## Author

Developed by **Erwan Ouabdesselam**  
Master’s in Applied Mathematics — Université Paris Dauphine, 2025

---

## License

This project is licensed under the [MIT License](LICENSE).
