#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:04:49 2018

@author: sayambhu
"""
import numpy as np
import pandas as pd

#The dataset . Make sure that the data set is in the same folder as this file


import pickle
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
import random
from sklearn.preprocessing import LabelBinarizer
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


text=pd.read_csv("mbti_1.csv" ,index_col='type')
snow = SnowballStemmer('english') 


def get_labels_1hot(text):
    labels=text.index.tolist()
    encoder=LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    labels=encoder.fit_transform(labels)
    labels=np.array(labels)
    return labels

def get_labels_4dim(text):
    labels=text.index.tolist()
    labels2=np.zeros([np.shape(labels)[0],4])
    for i in range(len(labels)):
        One=labels[i][0]
        Two=labels[i][1]
        Three=labels[i][2]
        Four=labels[i][3]
        if One=='E':
            labels2[i,0]=1
        else:
            labels2[i,0]=0
        if Two=='N':
            labels2[i,1]=1
        else:
            labels2[i,1]=0
        if Three=='T':
            labels2[i,2]=1
        else:
            labels2[i,2]=0
        if Four=='J':
            labels2[i,3]=1
        else:
            labels2[i,3]=0
    return labels2
    
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

#from autocorrect import spell

def save_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)



def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

 
def absurdities(Onefreqwords,word_count,post_list,freq,stem):
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
        for j in range(len(post_list[i])):
            try:
                ind=joint[post_list[i][j]]
            except:
                #print(i,j)
                if stem==1:
                    lis.append(snow.stem(post_list[i][j]))
                else:
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
    Onefreqwordsn1=[word for word in all_wordsn1 if all_wordsn1[word]==freq]
    
        
    return Onefreqwordsn1,all_wordsn1,post_list1
def cleanmess(text):
    posts=text.posts.tolist()
    posts=[post_cleaner(post) for post in posts]
    post_list=[word_tokenize(post) for post in posts]
    
    word_count=Counter()
    for post in posts:
        word_count.update(word_tokenize(post))
    
    vocab_len=len(word_count)
    Onefreqwords=[words for words in word_count if word_count[words]==1]
    Highfreqwords=[words for words in word_count if word_count[words]>1]
    
    Onefreqwordsn1,all_wordsn1,post_list1=absurdities(Onefreqwords,word_count,post_list,1,-1)
    
    Onefreqwordsn2,all_wordsn2,post_list2=absurdities(Onefreqwordsn1,all_wordsn1,post_list1,1,-1)
    
    Onefreqwordsn3,all_wordsn3,post_list3=absurdities(Onefreqwordsn2,all_wordsn2,post_list2,1,-1)
    Onefreqwordsn4,all_wordsn4,post_list4=absurdities(Onefreqwordsn3,all_wordsn3,post_list3,1,1)
    
    
    return Onefreqwordsn4,all_wordsn4,post_list4





def SeparationwithUNK(post_list,labels):
    X_train, X_test, y_train, y_test = train_test_split(post_list, labels, test_size=0.2, random_state=42)
    train_vocab=[]
    for i in range(len(X_train)):
        train_vocab.extend(X_train[i])

    train_vocab=Counter(train_vocab)
    Onefreqtrain=[x for x in train_vocab if train_vocab[x]==1]
    #Just in case this number becomes 1
    k=random.sample(range(len(Onefreqtrain)),min(1,int(len(Onefreqtrain))))
    Unknowns=[]
    for i in k:
        Unknowns.append(Onefreqtrain[i])
    print("Number of unknowns= ",len(Unknowns))
    
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
    
    return X_train,X_test,y_train,y_test,Xtr,Xtst,vocab


labels1hot=get_labels_1hot(text)
labels4dim=get_labels_4dim(text)
    
    
Onefreqwords,all_words,post_list=cleanmess(text)
ALLIs=[]
ALLEs=[]
ALLNs=[]
ALLSs=[]
ALLFs=[]
ALLTs=[]
ALLJs=[]
ALLPs=[]
for i in range(np.shape(labels1hot)[0]):
    if labels4dim[i,0]==1:
        ALLEs.extend(post_list[i])
    else:
        ALLIs.extend(post_list[i])
    if labels4dim[i,1]==1:
        ALLNs.extend(post_list[i])
    else:
        ALLSs.extend(post_list[i])
    if labels4dim[i,2]==1:
        ALLTs.extend(post_list[i])
    else:
        ALLFs.extend(post_list[i])
    if labels4dim[i,3]==1:
        ALLJs.extend(post_list[i])
    else:
        ALLPs.extend(post_list[i])

ALLIs=" ".join(ALLIs)
ALLEs=" ".join(ALLEs)
ALLNs=" ".join(ALLNs)
ALLSs=" ".join(ALLSs)
ALLFs=" ".join(ALLFs)
ALLTs=" ".join(ALLTs)
ALLJs=" ".join(ALLJs)
ALLPs=" ".join(ALLPs)


#X_train,X_test,y_train,y_test,Xtr,Xtst,vocab=SeparationwithUNK(post_list,labels1hot)
    
#X_train1,X_test1,y_train1,y_test1,Xtr1,Xtst1,vocab1=SeparationwithUNK(post_list,labels4dim)



# =============================================================================
# save_pickle(X_train,'X_train.pkl')
# save_pickle(y_train,'y_train.pkl')
# save_pickle(X_test,'X_test.pkl')
# save_pickle(y_test,'y_test.pkl')
# save_pickle(Xtr,'Xtr.pkl')
# save_pickle(Xtst,'Xtst.pkl')
# save_pickle(X_train1,'X_train1.pkl')
# save_pickle(y_train1,'y_train1.pkl')
# save_pickle(X_test1,'X_test1.pkl')
# save_pickle(y_test1,'y_test1.pkl')
# save_pickle(Xtr1,'Xtr1.pkl')
# save_pickle(Xtst1,'Xtst1.pkl')
# save_pickle(vocab,'vocab.pkl')
# save_pickle(vocab,'vocab1.pkl')
# =============================================================================
from os import path
from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS

    
wordcloud = WordCloud(
        background_color='black',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(ALLIs)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



    
wordcloud = WordCloud(
        background_color='black',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(ALLEs)
plt.imshow(wordcloud)
plt.axis("off")
plt.show() 



    
wordcloud = WordCloud(
        background_color='black',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(ALLNs)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



    
wordcloud = WordCloud(
        background_color='black',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(ALLSs)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


    
wordcloud = WordCloud(
        background_color='black',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(ALLFs)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


    
wordcloud = WordCloud(
        background_color='black',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(ALLTs)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



    
wordcloud = WordCloud(
        background_color='black',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(ALLJs)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



    
wordcloud = WordCloud(
        background_color='black',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(ALLPs)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
    
