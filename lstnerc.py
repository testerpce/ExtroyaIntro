#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 17:53:49 2018

@author: sayambhu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:16:42 2018

@author: sayambhu
"""
import numpy as np
import pickle
from keras.utils import plot_model
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



input = Input(shape=(mak,))
model = Embedding(input_dim=len(vocab_to_int)+1, output_dim=250,input_length=mak,
                   mask_zero=True)(input)  # 20-dim embedding
model = LSTM(units=400, return_sequences=False,
                           recurrent_dropout=0.1)(model) 
out=Dense(4,activation="sigmoid")(model)
model = Model(input, out)

model.summary()
plot_model(model,"LSTM.png")
from keras import metrics
model.compile(optimizer="Adam", loss="mean_squared_error", metrics=[metrics.mae,metrics.categorical_accuracy])

history = model.fit(Xtrc[0:100], y_trainc[0:100], batch_size=1, epochs=5,validation_split=0.05, verbose=1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
confE=[]
confN=[]
confT=[]
confJ=[]

#precal=[]

yactual=[]
ypred=[]
for i in range(np.shape(Xtstc)[0]-1500):
    print(i)
    y=model.predict(Xtstc[i].reshape(1,-1))
    One=int(y[0,0])
    Two=int(y[0,1])
    Three=int(y[0,2])
    Four=int(y[0,3])
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






