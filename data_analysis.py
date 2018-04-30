    # -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 21:41:15 2018

@author:Rishi
"""
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
import pickle

data = pd.read_csv('mbti_1.csv')
plt.figure(figsize=(40,20))
plt.xticks(fontsize=24, rotation=0)
plt.yticks(fontsize=24, rotation=0)
sns.countplot(data=data, x='type')


#============================= preprocessing ================================#

from sklearn.preprocessing import LabelEncoder
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
lab_encoder = LabelEncoder().fit(unique_type_list)


from nltk.corpus import stopwords 
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
cachedStopWords = stopwords.words("english")


def pre_process_data(data, remove_stop_words=True):

    list_personality = []
    list_posts = []
        
    for row in data.iterrows():
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        type_labelized = lab_encoder.transform([row[1].type])[0]
        list_personality.append(type_labelized)
        list_posts.append(temp)
       
    return list_posts, list_personality

list_posts, list_personality = pre_process_data(data, remove_stop_words=True)



#=========================== nltk pos tagging ===============================#

posts=[list()]*len(list_posts)
posts_tags=[list()]*len(list_posts)
for i in range(len(list_posts)):
    wordsList = nltk.word_tokenize(list_posts[i])
    tagged = nltk.pos_tag(wordsList)
    post=[[],[]]
    for a,b in tagged:
        post[0].append(a)
        post[1].append(b)
    posts[i]=post[0]
    posts_tags[i]=post[1]
del post,i,a,b,wordsList

''' dump the pos tags '''
with open('pos_tags.pckl', 'wb') as outfile:
    pickle.dump(posts_tags,outfile)
    outfile.close()
       

#============================= word2vec & glove ==============================#

gloveFile='./glove.6B/glove.6B.300d.txt.word2vec'


print("Loading Glove Model...")
f = open(gloveFile,'r',encoding='utf8')
model = {}
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding
print("Done.",len(model)," words loaded!")

# max post length is 923 tokens
max_document_length =  max([len(x.split(" ")) for x in list_posts])
# padding every post into 300*923 size glove embedding

X_glove=[list()]*len(list_posts)
for i in range(len(list_posts)):
    glove=np.zeros([300,923])
    j=0
    for word in list_posts[i].split(" "):
        if word in model:
            glove[:,j]= model[word] 
        j+=1
    X_glove[i]=glove

''' dump the glove model '''
with open('glove_model.pckl', 'wb') as outfile:
    pickle.dump(X_glove,outfile)
    outfile.close()
    
    

#=============================== Truncated SVD ===============================#

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

cntizer = CountVectorizer(analyzer="word", 
                             max_features=1500, 
                             tokenizer=None,    
                             preprocessor=None, 
                             stop_words=None,  
                             max_df=0.5,
                             min_df=0.1) 
                                 
tfizer = TfidfTransformer()

print("CountVectorizer")
X_cnt = cntizer.fit_transform(list_posts)
print("Tf-idf")
X_tfidf =  tfizer.fit_transform(X_cnt).toarray()

reverse_dic = {}
for key in cntizer.vocabulary_:
    reverse_dic[cntizer.vocabulary_[key]] = key
    
from sklearn.decomposition import TruncatedSVD
# Truncated SVD
svd = TruncatedSVD(n_components=12, n_iter=7, random_state=42)
svd_vec = svd.fit_transform(X_tfidf)

print("TSNE")
X_tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=650).fit_transform(svd_vec)

fig, ax = plt.subplots()
groups = pd.DataFrame(X_tsne[:,(0,2)], columns=['x', 'y']).assign(category=data.type).groupby('category')
col = list_personality
for name, points in groups:
    ax.scatter(points.x, points.y, label=name, cmap=plt.get_cmap('tab20') , s=12)
ax.legend()    
    

plt.figure(0)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=col, cmap=plt.get_cmap('tab20') , s=12)
plt.figure(1)
plt.scatter(X_tsne[:,0], X_tsne[:,2], c=col, cmap=plt.get_cmap('tab20') , s=12)
plt.figure(2)
plt.scatter(X_tsne[:,1], X_tsne[:,2], c=col, cmap=plt.get_cmap('tab20') , s=12)
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#============Split mbti personality into 4 letters and binarize===============#

titles = ["Extraversion (E) - Introversion (I)",
          "Sensation (S) - INtuition (N)",
          "Thinking (T) - Feeling (F)",
          "Judgement (J) - Perception (P)"] 

b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'F':0, 'T':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'F', 1:'T'}, {0:'J', 1:'P'}]

def translate_personality(personality):
    return [b_Pers[l] for l in personality]

def translate_back(personality):
    '''
    transform binary vector to mbti personality
    '''
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

list_personality_bin = np.array([translate_personality(p) for p in data.type])

# Plot
def plot_tsne(X, i):
    plt.figure(i, figsize=(30,20))
    plt.title(titles[i])
    plt.subplot(3,1,1)
    plt.scatter(X[:,0], X[:,1], c=list_personality_bin[:,i], cmap=plt.get_cmap('Set1'), s=25)
    plt.subplot(3,1,2)
    plt.scatter(X[:,0], X[:,2], c=list_personality_bin[:,i], cmap=plt.get_cmap('Set2'), s=25)
    plt.subplot(3,1,3)
    plt.scatter(X[:,1], X[:,2], c=list_personality_bin[:,i], cmap=plt.get_cmap('Set3'), s=25)


#======================== xgboost =======================#
    
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
le = LabelEncoder()

with open('pos_tags.pckl', 'rb') as f:
    x = np.array(pickle.load(f))

from random import sample
f = int(0.8*(len(data)))
indices = sample(range(len(data)),f)

train_data = x[indices]
Y_train = data.iloc[indices,0]
test_data = np.delete(x,indices)
Y_test= np.delete(np.array(data.type),indices)

# one hot encoding of the pos tags
one_hot_encode = np.zeros([len(train_data),1])
mlb= MultiLabelBinarizer()


l=[]
for i in range(len(train_data)):
    a=train_data[i]
    l.append(a)
print(mlb.fit_transform(l).shape)
one_hot_encode= np.concatenate((one_hot_encode,mlb.fit_transform(l)),axis=1)

l=[]
one_hot_encode_test = np.zeros([len(test_data),1])
for i in range(len(test_data)):
    a=test_data[i]
    l.append(a)
print(mlb.transform(l).shape)
one_hot_encode_test = np.concatenate((one_hot_encode_test,mlb.transform(l)),axis=1)


import xgboost as xgb
le = LabelEncoder()

Y_train=le.fit_transform(Y_train)
Y_test=le.transform(Y_test)

dtrain = xgb.DMatrix(one_hot_encode, label=Y_train)
dtest = xgb.DMatrix(one_hot_encode_test)

watchlist = [(dtrain, 'train')]

for seed in [1234]:
    param = {'max_depth':5, 
             'eta':0.2, 
             'silent':1, 
             'num_class':16,
             'objective':'multi:softmax',
             'eval_metric': "merror",
             'colsample_bytree': 0.7,
             'booster': "gbtree",
             'seed': seed
             }
    
    num_round = 400
    plst = param.items()
    # bst is the best model for XGBoost
    bst = xgb.train( plst, dtrain, num_round, watchlist )
    
ypred = bst.predict(dtest)

accuracy=0
for i in range(ypred.shape[0]):
    accuracy += int(ypred[i] == Y_test[i]) 
accuracy/ypred.shape[0]
