---
title: Topic Modeling on Surface Trivia Question Dataset
date: 2023-08-16
tags: NLP, text-classification
group: topic-modling
order: 1
---


I have worked with transformers a bit in the past but did not really account for my progress or do any real comparisons. I will be using the dataset that I assembled for a trivia app in Elixir that I built. There are a handful of categories each with a good amount of questions, and we will see that the transformers approach makes the process much easier.

---

```python
import pandas as pd
import numpy as np
import json
import re
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
len(data)
```

    4000

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)
```

    Batches:   0%|          | 0/125 [00:00<?, ?it/s]


```python
import umap.umap_ as umap
umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=5, 
                            metric='cosine').fit_transform(embeddings)
```

```python
import hdbscan
cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)
```


```python
import matplotlib.pyplot as plt

# Prepare data
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
plt.colorbar()
```




![png](/assets/images/topic-modeling/01_transformers.png#img-thumbnail)



```python
docs_df = pd.DataFrame(data_array, columns=["Doc"])
docs_df['Topic'] = cluster.labels_
docs_df['Doc_ID'] = range(len(docs_df))
docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
```


```python
docs_per_topic.head()
```


```python
docs_df['Topic'].value_counts()
```


    -1     1338
     44     203
     17     203
     21     194
     29     137
     19     102
     38      96
     11      95
     39      86
     9       75
     33      73
     13      71
     45      70
     14      70
     35      67
     4       66
     10      64
     41      64
     7       64
     12      56
     42      54
     43      47
     36      47
     15      44
     23      42
     5       40
     26      38
     30      36
     8       34
     24      34
     37      33
     6       29
     3       28
     27      28
     0       27
     34      27
     2       26
     32      25
     18      21
     20      20
     28      19
     1       19
     22      19
     31      18
     40      18
     16      17
     25      16
    Name: Topic, dtype: int64


```python
docs_df.head()
```


```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  
tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))
```


```python
def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
```


```python
top_n_words[21][:10]
```




    [('television', 0.25199257520408436),
     ('tv', 0.2272489906622773),
     ('shows', 0.17306120187797722),
     ('network', 0.0945769606612006),
     ('theme', 0.08517609400440743),
     ('company', 0.08181918094364625),
     ('producer', 0.07445065395128529),
     ('executive', 0.05932842541421376),
     ('broadcast', 0.057600770775431076),
     ('series', 0.0476476709374204)]



```python
from sklearn.metrics.pairwise import cosine_similarity
for i in range(20):
    # Calculate cosine similarity
    similarities = cosine_similarity(tf_idf.T)
    np.fill_diagonal(similarities, 0)

    # Extract label to merge into and from where
    topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
    topic_to_merge = topic_sizes.iloc[-1].Topic
    topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

    # Adjust topics
    docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
    old_topics = docs_df.sort_values("Topic").Topic.unique()
    map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
    docs_df.Topic = docs_df.Topic.map(map_topics)
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

    # Calculate new topic words
    m = len(data)
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)

topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
```


```python
top_n_words[51][:10]
```


    [('trump', 0.10467902584887456),
     ('president', 0.06072602666892069),
     ('2020', 0.03411193922678639),
     ('america', 0.03269916901768407),
     ('democratic', 0.032277888078611434),
     ('donald', 0.029263497562009327),
     ('democrats', 0.0268655653148694),
     ('election', 0.02609372385412432),
     ('presidential', 0.025912575696792114),
     ('bernie', 0.025237479236590536)]


```python
top_n_words[50][:10]
```

    [('don', 0.03970939022232294),
     ('people', 0.03339189616992504),
     ('anxiety', 0.03049151218674598),
     ('life', 0.023705264451986674),
     ('mental', 0.023679815071311398),
     ('doesn', 0.02318471421412793),
     ('disorder', 0.02080708641397244),
     ('need', 0.01934262579411308),
     ('like', 0.01924398264657584),
     ('just', 0.019145351423775627)]


