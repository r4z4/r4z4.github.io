---
title: TREC Dataset with EDA - Run 02
date: 2023-07-07
tags: trec, augmentation, NLP
group: trec
order: 3
---

Another benefit that I see in starting small and just using some basic datasets, is that is gives us a nice opportunity to explore data augmentation techniques. There are plenty of them out there of course, but focusing on some of the simpler ones can give us a way to expand our training set and hopefully get some better results our of our model. But of course, like most of these trials, even when it all goes wrong at least we can examine the results closely and gain a better understanding of why any issues occur.

With that said, we will start Run_02 by using one of these simple data augmentation techniques: Simple Synonym Replacement

### The Stolen Code:

```python
        ########################################################################
        # Synonym replacement
        # Replace n words in the sentence with synonyms from wordnet
        ########################################################################

        #for the first time you use wordnet
        #import nltk
        #nltk.download('wordnet')
        from nltk.corpus import wordnet 

        def synonym_replacement(words, n):
            new_words = words.copy()
            random_word_list = list(set([word for word in words if word not in stop_words]))
            random.shuffle(random_word_list)
            num_replaced = 0
            for random_word in random_word_list:
                synonyms = get_synonyms(random_word)
                if len(synonyms) >= 1:
                    synonym = random.choice(list(synonyms))
                    new_words = [synonym if word == random_word else word for word in new_words]
                    #print("replaced", random_word, "with", synonym)
                    num_replaced += 1
                if num_replaced >= n: #only replace up to n words
                    break

            #this is stupid but we need it, trust me
            sentence = ' '.join(new_words)
            new_words = sentence.split(' ')

            return new_words

        def get_synonyms(word):
            synonyms = set()
            for syn in wordnet.synsets(word): 
                for l in syn.lemmas(): 
                    synonym = l.name().replace("_", " ").replace("-", " ").lower()
                    synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                    synonyms.add(synonym) 
            if word in synonyms:
                synonyms.remove(word)
            return list(synonyms)
```

There was a little hangup here as well. Wordnet really seems to just work when it feels like it.