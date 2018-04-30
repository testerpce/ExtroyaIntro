#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 21:39:12 2018

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
#The four dimensional labels
labels2=text.index.tolist()
labels3=np.zeros([np.shape(labels)[0],4])
for i in range(len(labels2)):
    One=labels2[i][0]
    Two=labels2[i][1]
    Three=labels2[i][2]
    Four=labels2[i][3]
    if One=='E':
        labels3[i,0]=1
    else:
        labels3[i,0]=0
    if Two=='N':
        labels3[i,1]=1
    else:
        labels3[i,1]=0
    if Three=='T':
        labels3[i,2]=1
    else:
        labels3[i,2]=0
    if Four=='J':
        labels3[i,3]=1
    else:
        labels3[i,3]=0
    
    


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


#ok instead of jumping into that vocab to int directly try changing it so that
#First divide into train test sets
#Then try to change it into ints and all

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(post_list2, labels, test_size=0.2, random_state=42)

X_train1, X_test1, y_train1, y_test1 = train_test_split(post_list2, labels, test_size=0.2, random_state=42)

#Checking train and test vocabulary
train_vocab=[]
#test_vocab=[]
for i in range(len(X_train)):
    train_vocab.extend(X_train[i])
# =============================================================================
# for i in range(len(X_test)):
#     test_vocab.extend(X_test[i])
# test_vocab=Counter(test_vocab)
# =============================================================================
train_vocab=Counter(train_vocab)

Onefreqtrain=[x for x in train_vocab if train_vocab[x]==1]

k=random.sample(range(len(Onefreqtrain)),int(0.05*len(Onefreqtrain)))


Unknowns=[]
for i in k:
    Unknowns.append(Onefreqtrain[i])


for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        if X_train[i][j] in Unknowns:
            X_train[i][j]='<UNK>'

train_vocab=[]
#test_vocab=[]
for i in range(len(X_train)):
    train_vocab.extend(X_train[i])

train_vocab=Counter(train_vocab)

#Now replacing the test vocabulary in the same way.
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        if X_test[i][j] not in train_vocab:
            X_test[i][j]='<UNK>'
test_vocab=[]
for i in range(len(X_test)):
    test_vocab.extend(X_test[i])

test_vocab=Counter(test_vocab)
#
vocab = sorted(train_vocab, key=train_vocab.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
int_to_vocab={ii:word for ii,word in enumerate(vocab,1)}

Xtr=[[vocab_to_int[x] for x in posts]for posts in X_train]
Xtst=[[vocab_to_int[x] for x in posts]for posts in X_test]
Xtr1=[[vocab_to_int[x] for x in posts]for posts in X_train1]
Xtst1=[[vocab_to_int[x] for x in posts]for posts in X_test1]



save_pickle(X_train,'X_train.pkl')
save_pickle(y_train,'y_train.pkl')
save_pickle(X_test,'X_test.pkl')
save_pickle(y_test,'y_test.pkl')
save_pickle(Xtr,'Xtr.pkl')
save_pickle(Xtst,'Xtst.pkl')
save_pickle(X_train1,'X_train1.pkl')
save_pickle(y_train1,'y_train1.pkl')
save_pickle(X_test1,'X_test1.pkl')
save_pickle(y_test1,'y_test1.pkl')
save_pickle(Xtr1,'Xtr1.pkl')
save_pickle(Xtst1,'Xtst1.pkl')
save_pickle(train_vocab,'train_vocab.pkl')
save_pickle(vocab_to_int,'vocab_to_int.pkl')
save_pickle(int_to_vocab,'int_to_vocab.pkl')














