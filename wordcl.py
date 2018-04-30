#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 01:16:51 2018

@author: sayambhu
"""
# Simple WordCloud
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

import pickle
from nltk.stem import SnowballStemmer
from nltk import word_tokenize

from wordcloud import WordCloud, STOPWORDS

text = 'all your base are belong to us all of your base base base'
text=pd.read_csv("mbti_1.csv" ,index_col='type')
labels=text.index.tolist()
posts=text.posts.tolist()

labels3=np.zeros([np.shape(labels)[0],4])
for i in range(len(labels)):
    One=labels[i][0]
    Two=labels[i][1]
    Three=labels[i][2]
    Four=labels[i][3]
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

ALLIs=[]
ALLEs=[]
ALLNs=[]
ALLSs=[]
ALLFs=[]
ALLTs=[]
ALLJs=[]
ALLPs=[]



wordcloud = WordCloud(
        background_color='white',
        stopwords=STOPWORDS,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(ALLIs)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()