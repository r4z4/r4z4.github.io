---
title: TREC Dataset with EDA
date: 2023-06-29
tags: trec, augmentation, NLP, EDA
group: trec
order: 1
---

```python
import numpy as np
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from textwrap import wrap
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils
```

## Load Raw Data

There were a couple of steps that I took that I will spare us all the details of, but just to give you an idea here is a snippet of the data in its original form:

    ...
    ENTY:animal What predators exist on Antarctica ?
    DESC:manner How is energy created ?
    NUM:other What is the quantity of American soldiers still unaccounted for from the Vietnam war ?
    LOC:mount What was the highest mountain on earth before Mount Everest was discovered ?
    HUM:gr What Polynesian people inhabit New Zealand ?
    ...

& so I needed to perform some initial cleaning on the text data to transform it into this form, where we can pick up below:

    ...
    ENTY@@animal@@What predators exist on Antarctica@@?
    DESC@@manner@@How is energy created@@?
    NUM@@other@@What is the quantity of American soldiers still unaccounted for from the Vietnam war@@?
    LOC@@mount@@What was the highest mountain on earth before Mount Everest was discovered@@?
    HU@@gr@@What Polynesian people inhabit New Zealand@@?
    ...

Don't ask why I chose the delimiter.


```python
df1 = pd.read_csv('data/clean/processed/train_1000.txt', sep='@@')
df2 = pd.read_csv('data/clean/processed/train_2000.txt', sep='@@')
df3 = pd.read_csv('data/clean/processed/train_3000.txt', sep='@@')
df4 = pd.read_csv('data/clean/processed/train_4000.txt', sep='@@')
df5 = pd.read_csv('data/clean/processed/train_5500.txt', sep='@@')
df_test = pd.read_csv('data/clean/processed/test_100.txt', sep='@@')
```


```python
frames = [df1, df2, df3, df4, df5]

df = pd.concat(frames)
df.shape
```

    (15452, 4)


```python
df.head()
```


## Clean / Preprocess Data

---
There are several steps that we need to take here. Many of them will depend on the type of data that we have & our end goal, though, too. For certain types of text we may or may not be interested in numerical values, so we may strip those out with a function. Or maybe we need to keep any punctuation around. In most cases we will remove any punctuation, but the idea is to always be thinking about your data and how you may need to adapt it for your specific use case to get the most out of it.

---
Some other alterations that are included in this step but may seem a little different are the more advanced linguistic techniques of stemming and lemmatization. We won't get into particulars here but the same idea applies, in that if you are to use these methods it is always good to review just what they are doing and why they may or may not be needed for our case. With that in mind, let's take a look and see what we should do here.

First things first, we're keeping things simple and only interested in two columns.


```python
df = df.drop(['definition','punctuation'], axis='columns')
```

I have two functions in my utils.py that do some text cleaning using a combination of the methods mentioned above. Here is what each of those looks like and the corresponding output:

```python
def clean_text(text, ):

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = stem_text(text) # stemming
    text = remove_special_characters(text) # remove punctuation and symbols
    text = remove_stopwords(text) # remove stopwords
    #text.strip(' ') # strip whitespaces again?

    return text
```

```python
def normalize_text(s):
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    return s
```


```python
df['question_normalized'] = [utils.normalize_text(s) for s in df['question']]
```


```python
df['question_cleaned'] = [utils.clean_text(s) for s in df['question']]
```


```python
df.head(20)
```

#### Creating a Document Term Matrix


```python
df_grouped=df[['entity','question_cleaned']].groupby(by='entity').agg(lambda x:' '.join(x))
df_grouped.head()
```


```python
cv=CountVectorizer(analyzer='word')
data=cv.fit_transform(df_grouped['question_cleaned'])
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names_out())
df_dtm.index=df_grouped.index
df_dtm.head(6)
```


Just another good example of why doing these sometimes tedious tasks has value. Might want to come back and examine what exactly items like 000 and 000th are doing in the dataset. Might be indicative of larger issues, or just a one off that we need to drop. Also, I just want to make sure that my instinct of why 007 is in there holds true.


```python
# Function for generating word clouds
def generate_wordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap(title,60)),fontsize=13)
  plt.show()
  
# Transposing document term matrix
df_dtm=df_dtm.transpose()

# Plotting word cloud for each product
for index,product in enumerate(df_dtm.columns):
  generate_wordcloud(df_dtm[product].sort_values(ascending=False),product)
```

---

![png](/images/trec/eda/trec_eda_0.png#md-img)

---

---

![png](/images/trec/eda/trec_eda_1.png#md-img)

---


```python
doe_stems = [
    'Jim does like oranges',
    'Jim does not like oranges',
    "Jim doesn't like oranges", 
    'Jim doe like oranges'
    ]
results = [utils.clean_text(s) for s in doe_stems]
print(results)
```

    ['jim doe like orang', 'jim doe like orang', 'jim doe nt like orang', 'jim doe like orang']


I'll be honest I do not actually use WordClouds all that often in practice, but in this case I think it helps us quite a bit. We already know not to expect too much from this dataset, but at least so far we can see that it makes sense. If we were really digging in we would want to address the "doe" stem and maybe find a way to differentiate that between the two different sets. Or at least it might warrant just examining the data and seeing where it appears, and maybe a strategy will emerge from there.


```python
from textblob import TextBlob
df['polarity']=df['question_cleaned'].apply(lambda x:TextBlob(x).sentiment.polarity)
df.head()
```


```python
question_polarity_sorted=pd.DataFrame(df.groupby('entity')['polarity'].mean().sort_values(ascending=True))

plt.figure(figsize=(16,8))
plt.xlabel('Polarity')
plt.ylabel('Entities')
plt.title('Polarity of Different Question Entities from TREC Dataset')
polarity_graph=plt.barh(np.arange(len(question_polarity_sorted.index)),question_polarity_sorted['polarity'],color='orange',)

# Writing product names on bar
for bar,product in zip(polarity_graph,question_polarity_sorted.index):
  plt.text(0.005,bar.get_y()+bar.get_width(),'{}'.format(product),va='center',fontsize=11,color='black')

# Writing polarity values on graph
for bar,polarity in zip(polarity_graph,question_polarity_sorted['polarity']):
  plt.text(bar.get_width()+0.001,bar.get_y()+bar.get_width(),'%.3f'%polarity,va='center',fontsize=11,color='black')
  
plt.yticks([])
plt.show()
```

---

![png](/images/trec/eda/trec_eda_6.png#md-img)

---




