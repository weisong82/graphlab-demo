#-*- coding: UTF-8 -*-
import os
import graphlab
'''
Suppose our text data is currently arranged into a single file,
where each line of that file contains all of the text in a single document
Data:
+--------------------------------+
|               X1               |
+--------------------------------+
| alainconnes alain connes i ... |
| americannationalstandardsi ... |
'''
if os.path.exists('wikipedia_w16'):
    sf = graphlab.SFrame('wikipedia_w16')
else:
    sf = graphlab.SFrame.read_csv('https://static.turi.com/datasets/wikipedia/raw/w16.csv', header=False)
    sf.save('wikipedia_w16')

bow = graphlab.text_analytics.count_words(sf['X1'])

##Bag-of-words
print bow[0].keys()[:5]

##We can save this representation of the documents as another column of the original SFrame.
sf['bow'] = bow


'''
TF-IDF(word,document)=N(word,document)∗log(1/∑dN(word,d)))
where N(w, d) is the number of times word w occurs in document d.
 This transformation can be done to an SArray of dict type containing documents in bow-of-words format using tf_idf.
''' ''''''
#tfidf = graphlab.text_analytics.tf_idf(sf['bow'])
#sf['tfidf'] = tfidf


query = ['beatles', 'john', 'paul']

#bm25_scores = graphlab.text_analytics.bm25(sf['X1'], query)
#print bm25_scores


"""
Text cleaning


"""
docs = sf['bow'].dict_trim_by_values(2)

# Create a RareWordTrimmer transformer.
from graphlab.toolkits.feature_engineering import RareWordTrimmer
trimmer = RareWordTrimmer(threshold=2)

# Fit and transform the data.
transformed_sf = trimmer.fit_transform(sf)


##remove stop words in english
docs = docs.dict_trim_by_keys(graphlab.text_analytics.stopwords(), exclude=True)
#print "docs:"
#print docs

'''
Tokenization
'''
tokenized_docs = graphlab.SFrame()
tokenized_docs['tokens'] = graphlab.text_analytics.tokenize(sf['X1'])
print "tokenized_docs:"
print tokenized_docs


'''
Part of Speech Extraction  词形抽取
 adj 形容词
'''
parts_of_speech = graphlab.SFrame()
parts_of_speech['adjectives'] = graphlab.text_analytics.extract_parts_of_speech(sf['X1'],chosen_pos=[graphlab.text_analytics.PartOfSpeech.ADJ])
print "parts_of_speech:"+parts_of_speech


'''
Sentence Splitting
you may want a sentiment score for each sentence in a document. The following command accomplishes this for you:
'''
sentences = graphlab.SFrame()
sentences['sent'] = graphlab.text_analytics.split_by_sentence(sf['X1'])
print "sentences:"+sentences