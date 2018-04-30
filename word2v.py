#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 15:44:16 2018

@author: sayambhu
"""



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('mbti_1.csv')

plt.figure(figsize=(40,20))
plt.xticks(fontsize=24, rotation=0)
plt.yticks(fontsize=24, rotation=0)
sns.countplot(data=df, x='type')
plt.show()
import re
from nltk import word_tokenize

from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = './glove.6B/glove.6B.300d.txt'
word2vec_output_file = './glove.6B/glove.6B.300d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = './glove.6B/glove.6B.300d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)

Xvec=[]
size=300
eros=0
for user in X:
    some=[]
    for post in user:
        
        words=word_tokenize(post)
        V=np.zeros([300,1])
        for veci in words:
            veci=veci.lower()
            veci = re.sub('[^0-9a-zA-Z]+', '', veci)
            if not veci:
                continue
            k=0
            try:
                wordVector = model[veci].reshape((size,1))
            except KeyError:
                print("not found! ",  veci)
                eros+=1
                continue
            V+=np.reshape(np.matrix(model.wv[veci]),(300,1))
            k+=1
        some.append(V/k)
    Xvec.append(some)



