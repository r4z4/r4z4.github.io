---
title: GloVe Run 02
date: 2023-03-04
tags: GloVe, NLP, embeddings
group: glove
order: 2
---

Now we will start to augment our data in order to get a better understand of what kinds of parameters matter and how they all interact. The benefit of keeping things relatively simple will allow us to see a lot more of the details that may be hidden away in some of the more complex examples that you may come across.

---


```python
import numpy as np
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
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
df['subject'] = df['subject'].apply(lambda x: utils.replace_rejoin(x))
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


    (28245,
     9415,
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

    [[7186   20 4935    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [ 933   38 2247 1972    8   84  281    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [3365  641    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]]



    ((28245, 50), (9415, 50))


```python
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



### Approach 1: GloVe Embeddings Flattened (Max Tokens=50, Embedding Length=300) 

### Load previously trained model


```python
# Load model
model_file = 'models/sparse_cat_entire'
model = keras.models.load_model(model_file)
model._name = "SparseCategoricalEntireData"
model.summary()
```

    Model: "SparseCategoricalEntireData"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_2 (Embedding)     (None, 50, 50)            486750    
                                                                     
     flatten_2 (Flatten)         (None, 2500)              0         
                                                                     
     dense_6 (Dense)             (None, 128)               320128    
                                                                     
     dense_7 (Dense)             (None, 64)                8256      
                                                                     
     dense_8 (Dense)             (None, 7)                 455       
                                                                     
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
    883/883 [==============================] - 6s 6ms/step - loss: 0.6581 - accuracy: 0.8125 - val_loss: 0.5072 - val_accuracy: 0.8456
    Epoch 2/8
    883/883 [==============================] - 5s 6ms/step - loss: 0.3699 - accuracy: 0.8839 - val_loss: 0.4601 - val_accuracy: 0.8626
    Epoch 3/8
    883/883 [==============================] - 5s 6ms/step - loss: 0.2727 - accuracy: 0.9141 - val_loss: 0.4704 - val_accuracy: 0.8636
    Epoch 4/8
    883/883 [==============================] - 5s 6ms/step - loss: 0.2132 - accuracy: 0.9316 - val_loss: 0.4715 - val_accuracy: 0.8747
    Epoch 5/8
    883/883 [==============================] - 5s 6ms/step - loss: 0.1715 - accuracy: 0.9457 - val_loss: 0.4943 - val_accuracy: 0.8779
    Epoch 6/8
    883/883 [==============================] - 5s 6ms/step - loss: 0.1461 - accuracy: 0.9529 - val_loss: 0.5230 - val_accuracy: 0.8780
    Epoch 7/8
    883/883 [==============================] - 6s 6ms/step - loss: 0.1356 - accuracy: 0.9551 - val_loss: 0.5588 - val_accuracy: 0.8762
    Epoch 8/8
    883/883 [==============================] - 5s 6ms/step - loss: 0.1132 - accuracy: 0.9634 - val_loss: 0.5902 - val_accuracy: 0.8782


    <keras.callbacks.History at 0x7f5a97e5e0a0>


```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_preds = model.predict(X_test_vect).argmax(axis=-1)

print("Test Accuracy : {}".format(accuracy_score(y_test, y_preds)))
print("\nClassification Report : ")
print(classification_report(y_test, y_preds, target_names=target_categories))
print("\nConfusion Matrix : ")
print(confusion_matrix(y_test, y_preds))
```

    295/295 [==============================] - 1s 2ms/step
    Test Accuracy : 0.8781731279872543
    
    Classification Report : 
                  precision    recall  f1-score   support
    
           sport       0.83      0.86      0.85       992
           autos       0.87      0.92      0.90      3428
        religion       0.89      0.89      0.89      1312
       comp_elec       0.91      0.89      0.90      1212
         sci_med       0.90      0.84      0.87       988
          seller       0.76      0.71      0.74       486
        politics       0.93      0.83      0.88       997
    
        accuracy                           0.88      9415
       macro avg       0.87      0.85      0.86      9415
    weighted avg       0.88      0.88      0.88      9415
    
    
    Confusion Matrix : 
    [[ 857   70   14   11   15   12   13]
     [  70 3160   38   33   35   78   14]
     [  28   65 1165   36   10    3    5]
     [  13   61   41 1080    7    3    7]
     [  20   87   23   12  829    4   13]
     [  12  106    5    4    5  347    7]
     [  32   75   23   11   16   10  830]]



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
![png](/me/images/glove/run_02.png#img-thumbnail)

### Custom Test


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

    1/1 [==============================] - 0s 18ms/step


```python
print(doc_array)
```

    ['Democrats the Reuplicans are both the worst!' 'Coyotes win 10-0'
     'Houston Astros defeat the Cubs'
     'Apple and Microsoft both make great computers' 'New washer 4sale. $200']


```python
print(doc_array_vect)
```

    [[   1   48 2712    1  476    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [  52  465   65    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [7459 2362    1  988    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]
     [ 138    3  105 2712  279  402 1625    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0    0    0    0    0    0    0
         0    0    0    0    0    0    0    0]]


```python
print(cstm_test_preds)
```

    [6 1 6 1 3]


