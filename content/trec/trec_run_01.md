---
title: TREC Dataset with EDA - Run 01
date: 2023-07-06
tags: trec, augmentation, NLP
group: trec
order: 2
---

I guess a little introduction is warranted. I have always had a need to document my progress on anything. That used to be very scattered notes on paper, then scattered notes in various note-taking software formats. Having a living place to put down my thoughts just seemed like the natural progression, and that coupled with a rejuvenation of interest in model training and NLP in general – which lends itself to plenty of ever-changing concepts and the acronyms to go along with them – led me to this. This is my journey is trying to learn this stuff.  It is here to help me, but of course maybe some other lost soul will stumble across it.

### The Trial:

The first realization I think we all make when getting into this is the data out there is hard to come by, and from my experience that is only amplified when dealing with textual data. When you start getting into some of these large examples with extensive corpora then too, it just seems to muddy the waters. This is why I just wanted to start simple and struggle through some of these concepts again. So I chose a pretty simple dataset and just working with some basic text classification for now.

### The Idea:

Another thing I have noticed about myself is that having a concrete idea or goal in mind always helps. In this instance that just means coming up with some meaningless, contrived example that someone probably twenty years ago might’ve asked for as a product. So for some basic text classification, the idea is that a user would come to a site, being entering their input or submit their input, and the model will then classify that text and do whatever with it, and here we will just end up suggesting some simple template. Again, this will be completely contrived and trivial and may not look the prettiest.

### The Dataset:

TREC dataset contains 5500 labeled questions in training set and another 500 for test set. The dataset has 6 labels, 50 level-2 labels. Average length of each sentence is 10, vocabulary size of 8700.

---

Also we save it so we can reload and pickup training. Plenty of fighting with this finally settled on: seems best approach is to use the default `.save()` method for the TensorFlow SavedModel format (as opposed to a .keras or .h5 extension).
```python
model_file = 'models/5500'
model.save(model_file)
```
This approach actually creates a directory with asset files vs. a single file. To load, simple point to dir with `.load()`.
The NumPy method `assert_allclose()` also comes in handy for some extra reassurance with the saving & loading of the file.
```python
loaded_model = keras.models.load_model(model_file)
np.testing.assert_allclose(
    model.predict(validation_padded), loaded_model.predict(validation_padded)
)
```
--- 

```python
# fit model
num_epochs = 20
history = model.fit(train_padded, y_train, 
                    epochs=num_epochs, verbose=1,
                    validation_split=0.3)

# predict values
pred = model.predict(validation_padded)
```

And these are our results from run_01:
```
Epoch 1/20
271/271 [==============================] - 2s 6ms/step - loss: 1.7022 - accuracy: 0.2399 - val_loss: 1.6490 - val_accuracy: 0.2346
Epoch 2/20
271/271 [==============================] - 1s 3ms/step - loss: 1.6434 - accuracy: 0.2399 - val_loss: 1.6400 - val_accuracy: 0.2346
Epoch 3/20
271/271 [==============================] - 1s 3ms/step - loss: 1.6331 - accuracy: 0.2595 - val_loss: 1.6273 - val_accuracy: 0.2351
Epoch 4/20
271/271 [==============================] - 1s 3ms/step - loss: 1.6058 - accuracy: 0.3265 - val_loss: 1.5795 - val_accuracy: 0.3715
Epoch 5/20
271/271 [==============================] - 1s 3ms/step - loss: 1.5230 - accuracy: 0.3810 - val_loss: 1.4659 - val_accuracy: 0.3845
Epoch 6/20
271/271 [==============================] - 1s 3ms/step - loss: 1.3778 - accuracy: 0.4375 - val_loss: 1.3146 - val_accuracy: 0.4389
Epoch 7/20
271/271 [==============================] - 1s 4ms/step - loss: 1.2270 - accuracy: 0.5098 - val_loss: 1.1808 - val_accuracy: 0.5096
Epoch 8/20
271/271 [==============================] - 1s 3ms/step - loss: 1.0963 - accuracy: 0.5848 - val_loss: 1.0686 - val_accuracy: 0.6045
Epoch 9/20
271/271 [==============================] - 1s 3ms/step - loss: 0.9813 - accuracy: 0.6503 - val_loss: 0.9594 - val_accuracy: 0.6767
Epoch 10/20
271/271 [==============================] - 1s 3ms/step - loss: 0.8706 - accuracy: 0.7126 - val_loss: 0.8621 - val_accuracy: 0.7223
Epoch 11/20
271/271 [==============================] - 1s 4ms/step - loss: 0.7750 - accuracy: 0.7575 - val_loss: 0.7807 - val_accuracy: 0.7352
Epoch 12/20
271/271 [==============================] - 1s 3ms/step - loss: 0.6943 - accuracy: 0.7914 - val_loss: 0.7047 - val_accuracy: 0.7924
Epoch 13/20
...
271/271 [==============================] - 1s 4ms/step - loss: 0.3765 - accuracy: 0.8919 - val_loss: 0.4500 - val_accuracy: 0.8628
Epoch 20/20
271/271 [==============================] - 1s 3ms/step - loss: 0.3526 - accuracy: 0.8984 - val_loss: 0.4333 - val_accuracy: 0.8673
97/97 [==============================] - 0s 1ms/step
```
Now, it is easy to be fooled by the 89% and think we are great at this. Truth is this data is very insufficient and we have not done much to really challenge our model. As we go on to subsequent runs, though, we will see this score drop and vary and then come back up, which is reassuring. It does appear to be learning, but then the real question becomes how and why it is learning certain things in certain ways, how can we change this, and more importantly what can we do about it? What tools or insights can it give us, and from that, what value can we derive from it?

I will be printing and saving some simple visulaization images for later reference. I do not want to examine them all now without much to compare them too in these early stages, but they are certain valuable artifacts worth saving so that we can
revisit them later. I will just simply create a directory to store them for later viewing.

```python
print(pred)

    [[0.10461694 0.89111507 0.00184274 0.04174655 0.14603339 0.99627024]
    [0.02512768 0.07283066 0.14759105 0.98934025 0.1748144  0.80831176]
    [0.32353488 0.58661866 0.8057708  0.11234509 0.5822517  0.07421786]
    ...
    [0.2541242  0.75612414 0.49566412 0.08180005 0.30037883 0.2799908 ]
    [0.2660918  0.73733705 0.48640287 0.09529052 0.3022969  0.28448245]
    [0.40638423 0.8813672  0.9125333  0.02715247 0.2143805  0.03576801]]

print(y_test)

    [[0. 0. 0. 0. 0. 1.]
    [0. 0. 0. 1. 0. 0.]
    [0. 0. 1. 0. 0. 0.]
    ...
    [0. 0. 0. 0. 0. 1.]
    [0. 0. 0. 0. 0. 1.]
    [0. 0. 1. 0. 0. 0.]]
```
---


#### The dataset and documentation can be found [here](https://cogcomp.seas.upenn.edu/Data/QA/QC/)
#### You can also access the data via PyTorch. Details can be found [on PyTorch docs](https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.datasets.html)

```python
    torchnlp.datasets.trec_dataset(directory='data/trec/', train=False, test=False, train_filename='train_5500.label',
    test_filename='TREC_10.label', check_files=['train_5500.label'], urls=['http://cogcomp.org/Data/QA/QC/train_5500
    label', 'http://cogcomp.org/Data/QA/QC/TREC_10.label'], fine_grained=False)
```        

Let's see if it works with Notebook Output

---
```python
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding (Embedding)       (None, 120, 128)          1663616   
                                                                
global_average_pooling1d (G  (None, 128)              0         
lobalAveragePooling1D)                                          
                                                                
dense (Dense)               (None, 24)                3096      
                                                                
dense_1 (Dense)             (None, 6)                 150       
                                                                
=================================================================
Total params: 1,666,862
Trainable params: 1,666,862
Non-trainable params: 0
_________________________________________________________________
```


##### You can find the notebook [here](http://www.github.com/r4z4)

---

For the sake of brevity I'll just highlight some of the main steps, but all in all it is a pretty basic model. When we do a `mode.summary()` on it, we get:

```python
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding_1 (Embedding)     (None, 120, 16)           19200     
                                                                
global_average_pooling1d_1   (None, 16)               0         
(GlobalAveragePooling1D)                                        
                                                                
dense_2 (Dense)             (None, 24)                408       
                                                                
dense_3 (Dense)             (None, 6)                 150       
                                                                
=================================================================
Total params: 19,758
Trainable params: 19,758
Non-trainable params: 0
_________________________________________________________________
```

---

![png](/images/trec/run_01/run_01.png#md-img)

---