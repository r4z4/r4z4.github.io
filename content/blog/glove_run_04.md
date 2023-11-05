---
title: GloVe Run 04
date: 2023-03-07
tags: GloVe, NLP, embeddings
group: glove
order: 4
---

This will be the final run here with the GloVe embeddings and the last stage in the augmentation. So far we have not really seen to much improvement and I anticipate that will be the case here, but nevertheless let's go ahead and complete the cycle.

---

```python
import numpy as np
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import utils

import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
```


```python
df = pd.read_pickle('./data/dataframes/outer_merged_normalized_deduped.pkl')
df.head()
```

---


```python
all_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
# We'll use all
target_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
```

### Augment our data


```python
# But still need to fit the tokenizer on our original text to keep same vocak
max_tokens = 50 
X = np.array([subject for subject in df['subject']])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
```


```python
df['subject'] = df['subject'].apply(lambda x: utils.insert_rejoin(x))
```


```python
# container for sentences
X = np.array([subject for subject in df['subject']])
# container for sentences
y = np.array([subject for subject in df['newsgroup']])
```


```python
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(df['newsgroup'])
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)

classes = np.unique(y_train)
mapping = dict(zip(classes, target_categories))

len(X_train), len(X_test), classes, mapping
```


    (6359,
     2120,
     array([0, 1, 2, 3, 4, 5, 6]),
     {0: 'sport',
      1: 'autos',
      2: 'religion',
      3: 'comp_elec',
      4: 'sci_med',
      5: 'seller',
      6: 'politics'})


```python
## Vectorizing data to keep 50 words per sample.
X_train_vect = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_tokens, padding="post", truncating="post", value=0.)
X_test_vect  = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_tokens, padding="post", truncating="post", value=0.)

print(X_train_vect[:3])

X_train_vect.shape, X_test_vect.shape
```

    [[ 921  808   10 1301 1268    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [6123   19 1241    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [  26 8243 8244  239    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]]



    ((6359, 50), (2120, 50))


```python
# Should match previous runs
print("Vocab Size : {}".format(len(tokenizer.word_index)))
```

    Vocab Size : 9734

```python
path = './glove.6B.50d.txt'
glove_embeddings = {}
with open(path) as f:
    for line in f:
        try:
            line = line.split()
            glove_embeddings[line[0]] = np.array(line[1:], dtype=np.float32)
        except:
            continue
```


```python
embed_len = 50

word_embeddings = np.zeros((len(tokenizer.index_word)+1, embed_len))

for idx, word in tokenizer.index_word.items():
    word_embeddings[idx] = glove_embeddings.get(word, np.zeros(embed_len))
```


```python
word_embeddings[1][:10]
```


    array([ 0.15272   ,  0.36181   , -0.22168   ,  0.066051  ,  0.13029   ,
            0.37075001, -0.75874001, -0.44722   ,  0.22563   ,  0.10208   ])


---


### Approach 1: GloVe Embeddings Flattened (Max Tokens=50, Embedding Length=300) 

### Load previously trained model


---

```python
# Load model
model_file = 'models/sparse_cat_entire'
model = keras.models.load_model(model_file)
model.summary()
```

    Model: "SparseCategoricalEntireData"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, 50, 50)            486750    
                                                                     
     flatten (Flatten)           (None, 2500)              0         
                                                                     
     dense (Dense)               (None, 128)               320128    
                                                                     
     dense_1 (Dense)             (None, 64)                8256      
                                                                     
     dense_2 (Dense)             (None, 7)                 455       
                                                                     
    =================================================================
    Total params: 815,589
    Trainable params: 328,839
    Non-trainable params: 486,750
    _________________________________________________________________


```python
model.weights[0][1][:10], word_embeddings[1][:10]
```


    (<tf.Tensor: shape=(10,), dtype=float32, numpy=
     array([ 0.15272 ,  0.36181 , -0.22168 ,  0.066051,  0.13029 ,  0.37075 ,
            -0.75874 , -0.44722 ,  0.22563 ,  0.10208 ], dtype=float32)>,
     array([ 0.15272   ,  0.36181   , -0.22168   ,  0.066051  ,  0.13029   ,
             0.37075001, -0.75874001, -0.44722   ,  0.22563   ,  0.10208   ]))


```python
model.fit(X_train_vect, y_train, batch_size=32, epochs=8, validation_data=(X_test_vect, y_test))
```

    Epoch 1/8
    199/199 [==============================] - 2s 7ms/step - loss: 0.8248 - accuracy: 0.7789 - val_loss: 0.6934 - val_accuracy: 0.7896
    Epoch 2/8
    199/199 [==============================] - 1s 6ms/step - loss: 0.4073 - accuracy: 0.8800 - val_loss: 0.7078 - val_accuracy: 0.7953
    Epoch 3/8
    199/199 [==============================] - 1s 6ms/step - loss: 0.2645 - accuracy: 0.9291 - val_loss: 0.7498 - val_accuracy: 0.7929
    Epoch 4/8
    199/199 [==============================] - 1s 6ms/step - loss: 0.1771 - accuracy: 0.9530 - val_loss: 0.8251 - val_accuracy: 0.7816
    Epoch 5/8
    199/199 [==============================] - 2s 10ms/step - loss: 0.1199 - accuracy: 0.9734 - val_loss: 0.8772 - val_accuracy: 0.7788
    Epoch 6/8
    199/199 [==============================] - 2s 11ms/step - loss: 0.0815 - accuracy: 0.9847 - val_loss: 0.9525 - val_accuracy: 0.7722
    Epoch 7/8
    199/199 [==============================] - 2s 9ms/step - loss: 0.0632 - accuracy: 0.9888 - val_loss: 1.0143 - val_accuracy: 0.7764
    Epoch 8/8
    199/199 [==============================] - 1s 6ms/step - loss: 0.0497 - accuracy: 0.9904 - val_loss: 1.0631 - val_accuracy: 0.7726


    <keras.callbacks.History at 0x7f3cbd595130>


```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_preds = model.predict(X_test_vect).argmax(axis=-1)

print("Test Accuracy : {}".format(accuracy_score(y_test, y_preds)))
print("\nClassification Report : ")
print(classification_report(y_test, y_preds, target_names=target_categories))
print("\nConfusion Matrix : ")
print(confusion_matrix(y_test, y_preds))
```

    67/67 [==============================] - 1s 7ms/step
    Test Accuracy : 0.7726415094339623
    
    Classification Report : 
                  precision    recall  f1-score   support
    
           sport       0.65      0.59      0.62       189
           autos       0.83      0.88      0.85       901
        religion       0.73      0.65      0.69       220
       comp_elec       0.70      0.77      0.73       179
         sci_med       0.73      0.68      0.70       192
          seller       0.72      0.71      0.71       221
        politics       0.81      0.76      0.78       218
    
        accuracy                           0.77      2120
       macro avg       0.74      0.72      0.73      2120
    weighted avg       0.77      0.77      0.77      2120
    
    
    Confusion Matrix : 
    [[112  35  11   6   7  14   4]
     [ 19 794   6  15  24  26  17]
     [ 11  21 143  24   7   3  11]
     [  3  13  15 138   3   4   3]
     [  8  33   9   8 130   2   2]
     [ 13  44   2   2   3 156   1]
     [  6  18   9   4   4  12 165]]


```python
# TensorFlow SavedModel format => .keras
model_file = 'models/sparse_cat_entire'
model.save(model_file)
```


```python
!pip install scikit-plot
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
import numpy as np

skplt.metrics.plot_confusion_matrix([target_categories[i] for i in y_test], [target_categories[i] for i in y_preds],
                                    normalize=True,
                                    title="Confusion Matrix",
                                    cmap="Greens",
                                    hide_zeros=True,
                                    figsize=(5,5)
                                    );
plt.xticks(rotation=90);
```
---

![png](/images/glove/run_04.png#img-thumbnail)

### Custom Test

---


```python
# define documents
docs = ['Democrats the Reuplicans are both the worst!',
'Coyotes win 10-0',
'Houston Astros defeat the Cubs',
'Apple and Microsoft both make great computers',
'New washer 4sale. $200']

doc_array = np.array(docs)

doc_array_vect  = pad_sequences(tokenizer.texts_to_sequences(doc_array), maxlen=max_tokens, padding="post", truncating="post", value=0.)

cstm_test_preds = model.predict(doc_array_vect).argmax(axis=-1)
```

    1/1 [==============================] - 0s 179ms/step


```python
print(doc_array)
```

    ['Democrats the Reuplicans are both the worst!' 'Coyotes win 10-0'
     'Houston Astros defeat the Cubs'
     'Apple and Microsoft both make great computers' 'New washer 4sale. $200']


```python
print(doc_array_vect)
```

    [[   2   53 3716    2 1299    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [  69  199   47    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [2931 4257    2 1113    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [ 124    3  461 3716  277  580 1149    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [  27  842  542    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]]


```python
print(cstm_test_preds)
```

    [2 1 6 1 1]



