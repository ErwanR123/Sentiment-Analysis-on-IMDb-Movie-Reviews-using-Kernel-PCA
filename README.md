# Sentiment Analysis on IMDb Movie Reviews using Kernel PCA

Ce projet explore l'effet de la réduction de dimension non linéaire par **Kernel PCA** sur les performances de modèles classiques de classification dans une tâche de **détection de sentiment** à partir de critiques de films IMDb.

Le pipeline inclut :
- Prétraitement linguistique complet (lemmatisation, stopwords, POS-tagging)
- Vectorisation par **Bag-of-Words (CountVectorizer)** avec vocabulaire restreint
- Réduction de dimension par **Kernel PCA avec noyau cosinus**
- Évaluation de plusieurs modèles supervisés (KNN, LogReg, SVM)
- Analyse de sensibilité au nombre de composantes principales
- Comparaison avec les performances sans réduction de dimension

---

📘 **Remarque importante :**

> Le PDF fourni ci-dessous constitue une **version de travail** du rapport expérimental.  
> Il **n’est pas encore finalisé** et peut contenir quelques coquilles, erreurs mineures ou ajustements en cours.

👉 Consultez le document complet décrivant tout le processus et l'analyse des résultats ici :  
[`Impact_du_Kernel_PCA_sur_la_classification_de_sentiments.pdf`](./kernel_pca_sentiment_analysis.pdf
)

---

## References

- [IMDb Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Kernel PCA (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)

## Author

Developed by **Erwan Ouabdesselam** as part of my Master's thesis in Applied Mathematics, Université Paris-Dauphine, 2025.

## License

This project is licensed under the [MIT License](LICENSE).
