#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:28:27 2018

@author: sayambhu
"""


import numpy as np
import pandas as pd

import pickle
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
import random

#from autocorrect import spell

def save_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)



def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

snow = SnowballStemmer('english')  

#The dataset . Make sure that the data set is in the same folder as this file
text=pd.read_csv("mbti_1.csv" ,index_col='type')

#Making labels one hot
from sklearn.preprocessing import LabelBinarizer

labels=text.index.tolist()
encoder=LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
labels=encoder.fit_transform(labels)
labels=np.array(labels)

#Preprocessing data

import re

# Function to clean data ... will be useful later
def post_cleaner(post):
    """cleans individual posts`.
    Args:
        post-string
    Returns:
         cleaned up post`.
    """
    # Covert all uppercase characters to lower case
    post = post.lower() 
    
    # Remove |||
    post=post.replace('|||'," ") 
    


    # Remove URLs, links etc
    post = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', 'link', post, flags=re.MULTILINE) 
    # This would have removed most of the links but probably not all 
    post=re.sub('http',' ',post)

    # Remove puntuations 
    puncs1=['~','@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']','|','\\','"',"'",';',':','<','>','/']
    for punc in puncs1:
        post=post.replace(punc,'') 

    puncs2=[',','.','?','!','\n']
    for punc in puncs2:
        post=post.replace(punc,' ') 
    # Remove extra white spaces
    post=re.sub( '\s+', ' ', post ).strip()
    post=re.sub(r"(\w)\1{3,}", r'\1', post)#replace all repeated characters
    return post

# Clean up posts
# Covert pandas dataframe object to list. I prefer using lists for prepocessing. 
posts=text.posts.tolist()
posts=[post_cleaner(post) for post in posts]
post_list=[word_tokenize(post) for post in posts]

lens=[len(x) for x in post_list]

        
    
        

# Count total words
from collections import Counter
word_count=Counter()
for post in posts:
    word_count.update(word_tokenize(post))
    
vocab_len=len(word_count)
Onefreqwords=[words for words in word_count if word_count[words]==1]
Highfreqwords=[words for words in word_count if word_count[words]>1]

# Create a look up table 
# =============================================================================
# vocab = sorted(word_count, key=word_count.get, reverse=True)
# # Create your dictionary that maps vocab words to integers here
# vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
# =============================================================================

#Taking care of absurdities

joint=[]
breakup=[]
maxlength=[]

for word in Onefreqwords:
    test=word
    k=0
    current=[]
    while k==0:
        store=[]
        for i in range(len(word)):
            if test[0:i] in word_count:
                store.append(test[0:i])
        if len(store)!=0:
            test=re.sub(store[len(store)-1],'',test)
            maxlength.append(store[len(store)-1])
            current.append(store[len(store)-1])
        else:
            k=1
    joint.append(word)
    breakup.append(current)
joint={j:i for i,j in enumerate(joint)}
post_list1=[]
for i in range(len(post_list)):
    lis=[]
    for j in range(lens[i]):
        try:
            ind=joint[post_list[i][j]]
        except:
            #print(i,j)
            lis.append(post_list[i][j])
            continue
        #post_list[i][j:j+1]=breakup[ind]
        lis.extend(breakup[ind])
    post_list1.append(lis)
lens1=[len(x) for x in post_list1]
#all_wordsn=[]
all_wordsn1=[]
for i in range(len(post_list)):
    #all_wordsn+=post_list[i]
    all_wordsn1+=post_list1[i]
#Totalwordsn=all_wordsn
#all_wordsn=Counter(all_wordsn)
all_wordsn1=Counter(all_wordsn1)
lens1=[len(x) for x in post_list1]
#Onefreqwordsn=[word for word in all_wordsn if all_wordsn[word]==1]
Onefreqwordsn1=[word for word in all_wordsn1 if all_wordsn1[word]==1]



#absurdities2
joint=[]
breakup=[]
maxlength=[]

for word in Onefreqwordsn1:
    test=word
    k=0
    current=[]
    while k==0:
        store=[]
        for i in range(len(word)):
            if test[0:i] in word_count:
                store.append(test[0:i])
        if len(store)!=0:
            test=re.sub(store[len(store)-1],'',test)
            maxlength.append(store[len(store)-1])
            current.append(store[len(store)-1])
        else:
            k=1
    joint.append(word)
    breakup.append(current)
joint={j:i for i,j in enumerate(joint)}
post_list2=[]
for i in range(len(post_list1)):
    lis=[]
    for j in range(len(post_list1[i])):
        try:
            ind=joint[post_list1[i][j]]
        except:
            #print(i,j)
            lis.append(post_list1[i][j])
            continue
        #post_list[i][j:j+1]=breakup[ind]
        lis.extend(breakup[ind])
    post_list2.append(lis)
all_wordsn2=[]
for i in range(len(post_list2)):
    #all_wordsn+=post_list[i]
    all_wordsn2+=post_list2[i]
#Totalwordsn=all_wordsn
#all_wordsn=Counter(all_wordsn)
all_wordsn2=Counter(all_wordsn2)
lens2=[len(x) for x in post_list2]
#Onefreqwordsn=[word for word in all_wordsn if all_wordsn[word]==1]
Onefreqwordsn2=[word for word in all_wordsn2 if all_wordsn2[word]==1]


vocab = sorted(all_wordsn2, key=all_wordsn2.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

posts_ints=[]
for sent in post_list2:
    see=[]
    for word in sent:
        see.append(vocab_to_int[word])
    posts_ints.append(see)
see=[]
for i in range(len(posts_ints)):
    see+=posts_ints[i]

dick=Counter(see)
Uni=[word for word in dick if dick[word]==1]

posts_lens = Counter([len(x) for x in post_list2])
print("Zero-length reviews: {}".format(posts_lens[0]))
print("Maximum review length: {}".format(max(posts_lens)))
print("Minimum review length: {}".format(min(posts_lens)))

seq_len = 1000
features=np.zeros((len(posts_ints),seq_len),dtype=int)
for i, row in enumerate(posts_ints):
    if len(row)<seq_len:
        features[i, 0:len(row)] = np.array(row)[:len(row)]
    else:
        features[i, 0:seq_len] = np.array(row)[:seq_len]
print(features[:10])

split_frac = 0.9

num_ele=int(split_frac*len(features))
rem_ele=len(features)-num_ele
train_xA= features[:num_ele]
train_yA= labels[:num_ele]

test_xA =features[num_ele:]
test_yA = labels[num_ele:]

train_xNA=posts_ints[:num_ele]
train_yNA=train_yA
test_xNA =posts_ints[num_ele:]
test_yNA = test_yA

allws=[]
for sents in train_xA:
    allws+=sents.tolist()

Unique=Counter(allws)

Onefreq=[word for word in Unique if Unique[word]==1]

k=random.sample(range(len(Onefreq)),100)

Toreplace=np.array(Onefreq)[k]
vocab.append('<UNK>')
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}


for i in range(len(train_xA)):
    for j in range(len(train_xA[i])):
        
    
        



save_pickle(train_xA,'train_xA.pickle')
save_pickle(train_yA,'train_yA.pickle')
save_pickle(test_xA,'test_xA.pickle')
save_pickle(test_yA,'test_yA.pickle')
save_pickle(train_xNA,'train_xNA.pickle')
save_pickle(train_yNA,'train_yNA.pickle')
save_pickle(test_xNA,'test_xNA.pickle')
save_pickle(test_yNA,'test_yNA.pickle')













