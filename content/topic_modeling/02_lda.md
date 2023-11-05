---
title: Topic Modeling on Surface Trivia Question Dataset | Run 2 - Latent Dirichlet Allocation
date: 2023-04-09
tags: NLP, text-classification
group: topic-modling
order: 2
---

---

I wanted to go back and try a more traditional approach to the problem to have a comparison point for our transformers run. LDA is a tried
and true method for topic modeling, but as we will see, it requires a lot more effort than our previous example.

---

```python
import pandas as pd
import numpy as np
import json
import re
import gensim
```


```python
df = pd.read_json("trivia_data.json")
```


```python
## Gonna take a while to embed all the data (> 2hrs on CPU). Lets just use 20% of the data
# n = 20
# df = df.head(int(len(df)*(n/100)))

data = df['corrected_question']
data_array = np.array([e for e in df['corrected_question']])
```


```python
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

data_words = list(sent_to_words(data_array))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


    ['many', 'movies', 'stanley', 'kubrick', 'direct']



```python
import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])
```

    [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]



```python
from pprint import pprint
# number of topics
num_topics = 10
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```

    [(0,
      '0.017*"many" + 0.010*"name" + 0.009*"whose" + 0.007*"list" + '
      '0.006*"different" + 0.006*"school" + 0.006*"river" + 0.005*"also" + '
      '0.005*"common" + 0.005*"awards"'),
     (1,
      '0.018*"name" + 0.015*"whose" + 0.011*"many" + 0.011*"played" + 0.009*"also" '
      '+ 0.009*"team" + 0.008*"founded" + 0.007*"city" + 0.007*"sports" + '
      '0.005*"place"'),
     (2,
      '0.027*"name" + 0.018*"many" + 0.015*"whose" + 0.013*"people" + 0.010*"also" '
      '+ 0.007*"one" + 0.007*"person" + 0.007*"play" + 0.006*"city" + '
      '0.005*"company"'),
     (3,
      '0.017*"whose" + 0.013*"many" + 0.011*"team" + 0.009*"people" + '
      '0.008*"football" + 0.007*"also" + 0.007*"school" + 0.006*"list" + '
      '0.006*"place" + 0.006*"american"'),
     (4,
      '0.019*"show" + 0.016*"television" + 0.015*"whose" + 0.012*"name" + '
      '0.008*"list" + 0.006*"river" + 0.006*"many" + 0.006*"theme" + 0.005*"work" '
      '+ 0.005*"military"'),
     (5,
      '0.016*"also" + 0.015*"list" + 0.009*"team" + 0.008*"whose" + 0.008*"many" + '
      '0.006*"city" + 0.006*"people" + 0.006*"country" + 0.006*"teams" + '
      '0.005*"place"'),
     (6,
      '0.020*"many" + 0.016*"whose" + 0.013*"list" + 0.009*"also" + 0.008*"count" '
      '+ 0.008*"name" + 0.007*"company" + 0.006*"river" + 0.005*"one" + '
      '0.005*"awards"'),
     (7,
      '0.023*"name" + 0.014*"whose" + 0.011*"also" + 0.009*"many" + 0.009*"city" + '
      '0.009*"located" + 0.008*"team" + 0.006*"river" + 0.005*"state" + '
      '0.005*"country"'),
     (8,
      '0.030*"whose" + 0.015*"company" + 0.013*"list" + 0.011*"people" + '
      '0.009*"shows" + 0.009*"many" + 0.008*"name" + 0.007*"music" + 0.007*"team" '
      '+ 0.006*"count"'),
     (9,
      '0.023*"whose" + 0.016*"name" + 0.011*"one" + 0.010*"list" + 0.009*"also" + '
      '0.009*"people" + 0.009*"place" + 0.007*"many" + 0.007*"office" + '
      '0.006*"work"')]



```python
import pyLDAvis.gensim
import os
import pickle 
import pyLDAvis
# Visualize the topics
pyLDAvis.enable_notebook()
os.makedirs('./results', exist_ok=True)
LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+ str(num_topics) +'.html')
LDAvis_prepared
```

![png](/assets/images/topic-modeling/02_LDA.png#img-thumbnail)