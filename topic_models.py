# Download data if you haven't already
import graphlab as gl
import os
'''
"Topic models" are a class of statistical models for text data.
These models typically assume documents can be described by a small set of topics,
and there is a probability of any word occurring for a given "topic".

'''
if os.path.exists('wikipedia_w0'):
    docs = gl.SFrame('wikipedia_w0')
else:
    docs = gl.SFrame.read_csv('https://static.turi.com/datasets/wikipedia/raw/w0.csv', header=False)
    docs.save('wikipedia_w0')

# Remove stopwords and convert to bag of words
docs = gl.text_analytics.count_words(docs['X1'])
docs = docs.dict_trim_by_keys(gl.text_analytics.stopwords(), exclude=True)

# Learn topic model
model = gl.topic_model.create(docs)

'''
You may get details on a subset of topics by supplying a list of topic names or topic indices,
as well as restrict the number of words returned per topic.

'''
print model.get_topics()
'''
If we just want to see the top words per topic, this code snippet will rearrange the above SFrame to do that.

'''
print model.get_topics(output_type='topic_words')
