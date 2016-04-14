# Topic Models for Julia

Topic models are Bayesian, hierarchical mixture models of discrete data.  
This package implements utilities for reading and manipulating data commonly 
associated with topic models as well as inference and prediction procedures
for such models.

## Model description

The bulk of the package is designed for a particular topic model, Latent 
Dirichlet Allocation (LDA, Blei et al., 2003).  This model assumes a corpus
composed of a collection of bags of words; each bag of words is termed a
document.  The space whence the words are drawn is termed the lexicon.

Formally, the model is defined as

```
  For each topic k,
    phi_k ~ Dirichlet(beta)
  For each document d,
    theta ~ Dirichlet(alpha)
    For each word w,
      z ~ Multinomial(theta)
      w ~ Multinomial(phi_z)
```

alpha and beta are hyperparameters of the model.  The number of topics, K,
is a fixed parameter of the model, and w is observed.  This package fits 
the topics using collapsed Gibbs sampling (Griffiths and Steyvers, 2004).

## Package usage

We describe the functions of the package using an example. First we load 
corpora from data files as follows:

```
  testDocuments = readDocuments(open("cora.documents"))
  testLexicon = readLexicon(open("cora.lexicon"))
```

These read files in LDA-C format.  The lexicon file is assumed to have one
word per line.  The document file consists of one document per line.  Each
document consists of a collection of tuples; the first element of each tuple
expresses the word while the second element expresses the number of times
that word appears in the document.  The words are indicated by an index 
into the lexicon into the lexicon file, starting at zero.  The tuples are
separated by spaces and the entire line is prefixed by a number indicating
the number of tuples for that document.

With the documents loaded, we instantiate a model that we want to train:

```
  model = Model(fill(0.1, 10), 0.01, length(testLexicon), testDocuments)
```

This is a model with 10 topics.  alpha is set to a uniform Dirichlet prior
with 0.1 weight on each topic (the dimension of this variable is used
to determine the number of topics).  The second parameter indicates that
the prior weight on phi (i.e. beta) should be set to 0.01.  The third
parameter is the lexicon size; here we just use the lexicon we have 
just read.  The fourth parameter is the collection of documents.

```
  trainModel(testDocuments, model, 30)
```

With the model defined, we can train the model on a corpus of documents.
The trainModel command takes the corpus as the first argument, the model
as the second argument, and the number of iterations of collapsed Gibbs
samplign to perform as the third argument.  The contents of the model
will be mutated in place.

Finally we can examine the output of the trained model using topTopicWords.

```
  topWords = topTopicWords(model, testLexicon, 10)
```

This function retrieves the top words associated with each topic; this
serves as a useful summary of the model.  The first parameter is the model,
the second is the lexicon backing the corpus, and the third parameter
is the number of words to retrieve for each topic.  The output is an array
of arrays of the words in sorted order of prevalence in the topic.

## See also
The R package whence much of this code was derived at 
https://github.com/slycoder/R-lda-deprecated.
