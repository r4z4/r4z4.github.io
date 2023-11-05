---
title: TREC Dataset with EDA - Easy Data Augmentation - Methods
date: 2023-07-02
tags: trec, augmentation, NLP
group: trec
order: 4
---

This is another dataset where we have a relatively small dataset, and so we'll be using the EDA methods for some simple data augmentation, which will allow us to quickly and easily maximize the size of the data that we have. We will use the Synonym Repalcement, Random Insertion and Random Swap methods and see where that will take us.

```python
import pandas as pd
import random
```


```python
df = pd.read_pickle(r'data/dataframes/final_cleaned_normalized.pkl')
```


```python
df.head()
```


```python
import random
from random import shuffle
random.seed(1)

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line
```


```python
from nltk.corpus import wordnet
```


```python
########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

import nltk
nltk.download('wordnet')

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

def replace_rejoin_sr(x):
    words = synonym_replacement(x.split(), 1)
    sentence = ' '.join(words)
    return sentence
```

    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!



```python
replace_rejoin_sr('What is the total land mass of the continent of africa')
```




    'What is the add up land mass of the continent of africa'




```python
## Loop through and apply synonym_replacement for each headline
df['question_cleaned_sr'] = df['question_cleaned'].apply(lambda x: replace_rejoin_sr(x))
```


```python
df.head()
```



Note: I did need to add a check for word length - ```if len(words) > 1:``` - here -> Again, just a function of us using such a limited dataset. 


```python
########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		if len(words) > 1:
			add_word(new_words)
	return new_words

def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)
	
def replace_rejoin_ri(x):
    words = random_insertion(x.split(), 1)
    sentence = ' '.join(words)
    return sentence
```


```python
replace_rejoin_ri('What is the total land mass of the continent of africa')
```


    'What is the total represent land mass of the continent of africa'


```python
df['question_cleaned_ri'] = df['question_cleaned'].apply(lambda x: replace_rejoin_ri(x))
```


```python
########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		if len(words) > 1:
			new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

def replace_rejoin_rs(x):
    words = random_swap(x.split(), 1)
    sentence = ' '.join(words)
    return sentence
```


```python
replace_rejoin_rs('What is the total land mass of the continent of africa')
```


    'What is the total land mass africa the continent of of'


```python
df['question_cleaned_rs'] = df['question_cleaned'].apply(lambda x: replace_rejoin_rs(x))
```
---

For our particular case here we will not be using the Random Deletion. We can still perform the augmentation though and add it to our dataframe for reference, and I belive you will see why there is really no need for us to use this method here.

---

```python
########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

def replace_rejoin_rd(x):
    words = random_swap(x.split(), 1)
    sentence = ' '.join(words)
    return sentence
```


```python
replace_rejoin_rd('What is the total land mass of the continent of africa')
```


    'What the is total land mass of the continent of africa'


```python
df['question_cleaned_rd'] = df['question_cleaned'].apply(lambda x: replace_rejoin_rs(x))
```


```python
df.tail(4)
```

|    | entity      | question  | question_normalized   | question_cleaned | question_normalized   | question_cleaned | question_normalized   | question_cleaned |
|---:|:------------|:----------|-----------------------|------------------|-----------------------|------------------|-----------------------|------------------|
|  5432 | HUM   | What English explorer discovered and named Vir...| what english explorer discovered and named vir... | english explor discov name virginia  | english explor discov refer virginia | english explor discov va name virginia | english virginia discov name explor | explor english discov name virginia |
|  5433 | ENTY     | What war added jeep and quisling to the Englis...| what war added jeep and quisling to the englis...| war ad jeep quisl english languag  | warfare ad jeep quisl english languag |war ad jeep quisl english people english languag | ad war jeep quisl english languag | war ad jeep english quisl languag |
|  5434 | LOC   | What country is home to Heineken beer| what country is home to heineken beer | countri home heineken beer  |countri dwelling house heineken beer| countri home heineken rest home beer | countri home heineken beer | countri home beer heineken |
|  5435 | HUM   | What people make up half the Soviet Union 's p...| what people make up half the soviet union s po... | peopl make half soviet union popul |peopl realise half soviet union popul| peopl make north half soviet union popul | peopl popul half soviet union make | make peopl half soviet union popul |

---

Keeping with the theme of staying simple and concise, to augment our data - since we have a relatively very small dataset - we will turn to some simple techniques that were highlighted in a popular 2019 paper titled ["EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"](https://arxiv.org/abs/1901.11196). In the paper they introduce four simple techniques to performing data augmentation, and we will utilize them all for our dataset.

One very important point to bring up is the attention paid to the issue of how much augmentation to apply. For our purposes here, we are mainly just exploring in order to get a sense of what the techniques do and how they can - in general - affect our data. If we were engaging with real data for real business solutions, it is important to test a variety of sample sizes and tune with various hyperparameters. There is a large section in the paper dedicated to the question of how many sentences or items (naug) to augment, and note that researchers promote trying several out.

>  "For smaller training sets, over-fitting was more likely, so generating many augmented sentences yielded large performance boosts. For larger training sets, adding more than four augmented sentences per original sentence was unhelpful since models tend to generalize properly when large quantities of real data are available. (pg. 4)"

### Table 3: Recommended usage parameters.
---

| Ntrain | &nbsp; α       | &nbsp; naug |
| :---   |  :----:        | ---:        |
| 500    | &nbsp; 0.05    | &nbsp; 16   |
| 2,000  | &nbsp; 0.05    | &nbsp; 8    |
| 5,000  | &nbsp; 0.1     | &nbsp; 4    |
| More   | &nbsp; 0.1     | &nbsp; 4    |

---
#### Table 1: Sentences generated using EDA. SR: synonym replacement. RI: random insertion. RS: random swap. RD: random deletion.
---

| Operation | &nbsp; Sentence                                                                      |
| :---      | :---                                                                          |
| None      | &nbsp; A sad, superior human comedy played out on the back roads of life.            |
| SR        | &nbsp; A lamentable, superior human comedy played out on the backward road of life.  |
| RI        | &nbsp; A sad, superior human comedy played out on funniness the back roads of life.  |
| RS        | &nbsp; A sad, superior human comedy played out on roads back the of life.            |
| RD        | &nbsp; A sad, superior human out on the roads of life.                               |

---
