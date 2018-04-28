#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:14:53 2018

@author: aniketpramanick
"""
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed
from keras.layers import Bidirectional, concatenate, SpatialDropout1D
from keras.utils import plot_model
from keras.callbacks import Callback
import keras.backend as K
import pandas as pd
import numpy as np
import re, collections
import numpy as np
from random import sample
import nltk
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
import pickle as pkl
from sklearn.preprocessing import LabelEncoder

#List of Hypermeters

max_sent_len = 1000
num_sentences = 5000
EPOCHS = 5
BATCH = 100













unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ', '#####']
data = pd.read_csv("mbti_1.csv")
#print data["type"]



def pre_process_data(data, num_sentences, remove_stop_words=True):
    print("Gathering Data .....")
    tag_list = data["type"]
    post_list = data["posts"]
    print("Data Gathered .....")
    unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    lab_encoder = LabelEncoder().fit(unique_type_list)
    list_personality = []
    list_posts = []
    print("Removing Noise .....")
    for i in range(num_sentences):
    #for row in data.iterrows():
        #row = data.iterrows()[i]
        #posts = row[1].posts
        posts = post_list[i]
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        type_labelized = lab_encoder.transform([data["type"][i]])[0]
        list_personality.append(type_labelized)
        list_posts.append(temp)
        sys.stdout.write('\r[{}/{}]'.format(i+1, num_sentences))
    #print "\n"
    print("\nNoise Removed .....\n")
    return list_posts, list_personality

#list_posts, list_personality = pre_process_data(data, 150, remove_stop_words=True)

#print list_posts
#print list_personality

def tagger(data, num_sentences, remove_stop_words = True):
    list_posts, list_personality = pre_process_data(data, num_sentences, remove_stop_words=True)
    post_list = list()
    print("POS tagging data .....")
    for i in range(len(list_posts)):
        post_list.append(nltk.pos_tag(nltk.word_tokenize(str(list_posts[i]).decode("ascii", "ignore"))))
        sys.stdout.write('\r[{}/{}]'.format(i+1, len(list_posts)))
    #print "\n"
    print("\nData Tagged with POS tags .....\n")
    return post_list, list_personality


def word_list(pos_tagged_data):
    word_list = [tup[0] for sentence in pos_tagged_data for tup in sentence]
    #padded_data = pad_data(data)
    return word_list

def pos_list(pos_tagged_data):
    pos_list = [tup[1] for sentence in pos_tagged_data for tup in sentence]
    #padded_data = pad_data(data)
    return pos_list

def avg_sent_len(data):
    length = 0
    #padded_data = pad_data(data)
    for sentence in data:
        length += len(sentence)
    return int(float(length)/len(data)+1)


def create_vocab(pos_tagged_data):
    #padded_data = pad_data(data)
    print("Creating Vocabulary .....")
    wordlist = word_list(pos_tagged_data)
    poslist = pos_list(pos_tagged_data)
    freq_dict_words = collections.Counter(wordlist)
    freq_dict_pos = collections.Counter(poslist)
    least_occurred_words = [word for word in freq_dict_words.keys() if freq_dict_words[word] <= 1]
    least_occurred_pos = [pos for pos in freq_dict_pos.keys() if freq_dict_pos[pos] <= 1]
    #print 0.05*len(least_occurred_words)
    number_words_deleted = int(0.05*len(least_occurred_words))
    #number_pos_deleted = int(0.05*len(least_occurred_pos))
    for i in range(number_words_deleted):
        del freq_dict_words[least_occurred_words[i]]
        freq_dict_words['<UNK>'] += 1
    
    freq_dict_pos['<UNK>'] += 1
    #Word and frequency tuple sorted in descending order.
    word_freq_tuple = sorted(freq_dict_words.items(), key = lambda element: (-element[1], element[0]))
    pos_freq_tuple = sorted(freq_dict_pos.items(), key = lambda element: (-element[1], element[0]))
    words, word_frequency = list(zip(*word_freq_tuple))
    pos, pos_frequency = list(zip(*pos_freq_tuple))
    #print words.index("<UNK>"), freq_dict['<UNK>']
    #Creating Reverse word list i.e. index is the value and key is the word.
    word_to_id = dict(zip(words, range(len(words))))
    pos_to_id = dict(zip(pos, range(len(pos))))
    reverse_dictionary_words = dict(zip(word_to_id.values(), word_to_id.keys()))
    reverse_dictionary_pos = dict(zip(pos_to_id.values(), pos_to_id.keys()))
    #print word_to_id['<UNK>']
    print("Vocabulary Created .....")
    return reverse_dictionary_words, reverse_dictionary_pos ,word_to_id, pos_to_id

def max_sent(pos_tagged_data):
    s = -1
    for sentence in pos_tagged_data:
        if len(sentence)>s:
            s = len(sentence)
    return s


#max_sent_len = 1000

post_list, list_personality = tagger(data, num_sentences, True)
reverse_dictionary_words, reverse_dictionary_pos ,word_to_id, pos_to_id = create_vocab(post_list)
#print post_list

#print max_sent(post_list)
print("Generating Input and Output Vectors .....")
X_word = [[word_to_id[word[0]] if word[0] in word_to_id else word_to_id['<UNK>'] for word in sentence] for sentence in post_list]
X_pos = [[pos_to_id[word[1]] if word[1] in pos_to_id else pos_to_id['<UNK>'] for word in sentence] for sentence in post_list]
Y = list()
for i in range(len(post_list)):
    Y.append([list_personality[i]]*len(post_list[i]))


    
print("Input and output Vectors Generated .....")
print("Vector Preprocessing Started .....")

X_word = pad_sequences(maxlen = max_sent_len, sequences = X_word, padding = "post", value = word_to_id["<UNK>"])
X_pos = pad_sequences(maxlen = max_sent_len, sequences = X_pos, padding = "post", value = pos_to_id["<UNK>"])
Y = pad_sequences(maxlen = max_sent_len, sequences = Y, padding = "post", value = 17)  



print("Data Vectors Modified .....")
print("Converting into one-hot representations .....")
print("One-hot vectors generated .....")   


input_data = Input(shape = (max_sent_len, ))
word_embedding = Embedding(input_dim = len(word_to_id.keys()), output_dim = 25, input_length = max_sent_len, mask_zero = True)(input_data)
input_pos = Input(shape = (max_sent_len,  ))
pos_embedding = Embedding(input_dim = len(pos_to_id.keys()), output_dim = 5, input_length = max_sent_len, mask_zero = True)(input_pos)

X = concatenate([word_embedding, pos_embedding])
print("Input Embedded .....")
X = SpatialDropout1D(0.1)(X)
lstm = Bidirectional(LSTM(units = 50, return_sequences = True, recurrent_dropout=0.3))(X)
out = TimeDistributed(Dense(len(unique_type_list)+1, activation = 'sigmoid'))(lstm)
model = Model([input_data, input_pos], out)
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.summary()
plot_model(model, to_file='attention_model.png')
history = model.fit([X_word, np.array(X_pos).reshape((len(X_pos), max_sent_len))], np.array(Y).reshape(len(Y), max_sent_len, 1), batch_size = BATCH, epochs = EPOCHS, validation_split = 0.1, verbose = 1)
loss, accuracy = model.evaluate([X_word, np.array(X_pos).reshape((len(X_pos), max_sent_len))], np.array(Y).reshape(len(Y), max_sent_len, 1), batch_size= BATCH, verbose=1)
print("\nTest Loss: {}, Accuracy on Test Data: {} \n".format(loss, accuracy))
histogram = pd.DataFrame(history.history)


plt.style.use("ggplot")
plt.figure(1)
plt.plot(histogram["acc"], label = "Test Accuracy")
plt.plot(histogram["val_acc"], label = "Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc = "upper left")
plt.savefig("accuracy.png")
plt.show()