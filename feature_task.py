import graphlab as gl
# Construct a transformer
sf = gl.SFrame({'docs': ["This is a document!", "This one's also a document."]})
f = gl.feature_engineering.TFIDF(features = ['docs'])

# Fit it to a dataset
f.fit(sf)

# Now the object is ready to transform new data
f.transform(sf)



"""
 Numeric Features

Quadratic Features
Feature Binning
Numeric Imputer
Categorical Features

One Hot Encoder
Count Thresholder
Categorical Imputer
Count Featurizer
Image Features

Deep Feature Extractor
Text features

TF-IDF
Tokenizer
RareWordTrimmer
BM25
Misc.

Hasher
Random Projection
Transformer Chain
Custom Transformer

"""