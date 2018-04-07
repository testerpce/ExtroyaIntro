# ExtroyaIntro
NLU project

Let it Go! Let it goooo!

Files in this repository(Update as it is put in)

1)preprocess.py (Contains the preprocessed text data and the batches for training, validation and test sets




I am adding the preprocessing file for now.


# One improvement all of us should try to implement is using 4 separate  two class classifiers. Since each label has a label of 4 tags of which each tag can be one of two entities so Each of 4 classifiers will try to predict one entity for each tag  e.g. for _,_,_,_ each position is a tag and here the first tag takes values 'E' or 'N'

I will upload the preprocessing file later where there will be four binary columns indicating one of the two binary entities for each of four tags.
Update(sayambhu): I have now added the preprocessing where the four different binary columns for four different label types are given.
#E for 1 and I for 0
#N for 1 and S for 0
#T for 1 and F for 0
#J for 1 and P for 0


# Edit: I have uploaded the pickle files and I made a mistake. Each row is a 500 length vector


Aniket will try to implement the RNN model in the kernel for that facebook data.
#The improvement point is using biRNNs. Try to do the same code using biRNNs

https://www.kaggle.com/prnvk05/rnn-mbti-predictor/notebook

I will try to implement the Bag of words based classifier according to this kernel
#I know there is no improvement on using these methods which have tfidf or bag of words.
# But mostly I will improve preprocessing as much as possible. You two should not focus on preprocessing. I will do that.

https://www.kaggle.com/depture/multiclass-and-multi-output-classification

Rishi will try to implement the CNN based classifier according to this code on github or his own code if he wants
#The improvement part might be to get multilayer CNNs or different pooling methods

https://github.com/aboyker/convnet-document-classification




