#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:16:42 2018

@author: sayambhu
"""

import pickle

def save_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


Xtr=load_pkl('Xtr1.pkl')
Xtst=load_pkl('Xtst1.pkl')
y_train=load_pkl('y_train.pkl')
y_test=load_pkl('y_test.pkl')
vocab=load_pkl('vocab.pkl')
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

mak=1500
from keras.preprocessing.sequence import pad_sequences
Xtr = pad_sequences(maxlen=mak, sequences=Xtr, padding="post", value=vocab_to_int['<UNK>'])



from keras.models import Model, Input

from keras.layers import LSTM, Dense, Embedding,TimeDistributed, Dropout, Bidirectional



input = Input(shape=(mak,))
model = Embedding(input_dim=len(vocab_to_int)+1, output_dim=250,input_length=mak,
                   mask_zero=True)(input)  # 20-dim embedding
model = LSTM(units=256, return_sequences=False,
                           recurrent_dropout=0.1)(model) 
out=Dense(4,activation="softmax")(model)
model = Model(input, out)

model.summary()
from keras import metrics
model.compile(optimizer="Adam", loss="mean_squared_error", metrics=[metrics.mae,metrics.categorical_accuracy])

history = model.fit(Xtr, y_train, batch_size=20, epochs=5,validation_split=0.05, verbose=1)


