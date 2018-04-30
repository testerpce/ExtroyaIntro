#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:58:25 2018

@author: sayambhu
"""

import numpy as np
import pickle

def save_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

Xtrc=load_pkl('Xtrc1.pkl')
Xtstc=load_pkl('Xtstc1.pkl')
y_trainc=load_pkl('y_trainc1.pkl')
y_testc=load_pkl('y_testc1.pkl')
vocabc1=load_pkl('vocabc1.pkl')
vocab_to_int = {word: ii for ii, word in enumerate(vocabc1, 1)}

mak=8000
from keras.preprocessing.sequence import pad_sequences
Xtrc = pad_sequences(maxlen=mak, sequences=Xtrc, padding="post", value=vocab_to_int['`'])
Xtstc = pad_sequences(maxlen=mak, sequences=Xtstc, padding="post", value=vocab_to_int['`'])



from keras.models import Model, Input

from keras.layers import LSTM, Dense, Embedding,TimeDistributed, Dropout, Bidirectional

import keras.backend as K
from keras.layers import Lambda

from keras import metrics
re = Lambda(lambda x: K.reshape(x, (1, 1, 100)))


input = Input(shape=(mak,))
model = Embedding(input_dim=len(vocab_to_int), output_dim=250,
                  input_length=mak, mask_zero=True)(input)  # 20-dim embedding
model1 = Bidirectional(LSTM(units=50, return_sequences=False,
                           recurrent_dropout=0.1))(model)
output1 = Dense(1, activation='sigmoid', name='output1')(model1)




model2 = Bidirectional(LSTM(units=50, return_sequences=False,
                           recurrent_dropout=0.1))(re(model1))

output2 = Dense(1, activation='sigmoid', name='output2')(model2)

model3 = Bidirectional(LSTM(units=50, return_sequences=False,
                           recurrent_dropout=0.1))(re(model2))

output3 = Dense(1, activation='sigmoid', name='output3')(model3)
model4 = Bidirectional(LSTM(units=50, return_sequences=False,
                           recurrent_dropout=0.1))(re(model3))

output4 = Dense(1, activation='sigmoid', name='output4')(re(model4))

model = Model(inputs=input, outputs=[output1, output2, output3, output4])
model.compile(optimizer="Adam", loss="mean_squared_error", metrics=[metrics.mae,metrics.binary_accuracy])

model.summary()

y1=y_trainc
y2=y_trainc.reshape(1,-1).T
y3=y_trainc.reshape(1,-1).T
y4=y_trainc.reshape(-1,1,1)


history = model.fit(Xtrc, [y4,y2,y3,y1], batch_size=1, epochs=2,validation_split=0.05, verbose=1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
confE=[]
confN=[]
confT=[]
confJ=[]

#precal=[]

yactual=[]
ypred=[]
for i in range(np.shape(Xtstc)[0]):
    print(i)
    y=model.predict(Xtstc[i].reshape(1,-1))
    One=int(np.round(y[0]))
    Two=int(np.round(y[1]))
    Three=int(np.round(y[2]))
    Four=int(np.round(y[3]))
    ya=y_testc[i]
    Oneac=int(ya[3])
    Twoac=int(ya[1])
    Threeac=int(ya[2])
    Fourac=int(ya[0])
    ypre=[One,Two,Three,Four]
    yac=[Oneac,Twoac,Threeac,Fourac]
    ypred.append(ypre)
    yactual.append(yac)
    confE.append(confusion_matrix([Four],[Fourac]))
    confN.append(confusion_matrix([Two],[Twoac]))
    confT.append(confusion_matrix([Three],[Threeac]))
    confJ.append(confusion_matrix([One],[Oneac]))
    
    
confusionE=np.sum(confE,axis=0)  
confusionN=np.sum(confN,axis=0) 
confusionT=np.sum(confT,axis=0) 
confusionJ=np.sum(confJ,axis=0)  





