# NLP-Search-Engine

This repo contain my project developed during the "Natural Language Processing" course at university

The aim of the project is develope a NLP Search Engine, using **NLTK (Natural Language toolkit)** that given a query string the engine retrive the first $k$ documents in the corpus that have best similarity respect to the query, in this project I have explored the main tools useful in the NLP context, such as:

* Corpus loading
* Preprocessing on text data
  * Stopwords removal
  * Lemmatization
  * Tokenization
  * Punctuation removal
  * Part of Speech
  * Data cleaning in general
* Document representation 
  * Continous Bags of Word (CBOW)
  * Word embeddings (`Word2Vec`)
* Document represtation 
  * Embedding average for documents representation
  * `Doc2Vec` model
  * TF-IDF
* Cosine similarity
* K-means alghoritm
* t-SNE dimensionality reduction
* Evaluation of the model (`Precision`, `Recall`, `F1`)
* Spelling correction (using Levenshtein edit distance)

## Dependencies

- [`re`](https://docs.python.org/3/library/re.html)
- [`numpy`](https://numpy.org/)
- [`matplotlib`](https://matplotlib.org/)
- [`sklearn`](https://scikit-learn.org/stable/)
- [`nltk`](https://www.nltk.org/)
- [`gensim`](https://radimrehurek.com/gensim/)
- [`IPyWidgets`](https://ipywidgets.readthedocs.io/en/stable/)
- [Spelling Correction (more on my GitHub)](https://github.com/EmilioGarzia/Spelling-Correction/tree/master)

You can install theese dependencies from `requirements.txt` using `pip` manager in your environment as shown below:

```bash
pip install -r requirements.txt
```

## Author

[Emilio Garzia](https://github.com/EmilioGarzia), 2024
