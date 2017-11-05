# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:20:22 2017

@author: t
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:47:53 2017

@author: t
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset, source should be specified
path =  "rC:\\Users\\t\\.spyder-py3\\Python-Datasets\'show_reviews.tsv', delimiter = '\t', quoting =3)" 
 
dataset= pd.read_csv(r"C:\Users\t\.spyder-py3\Python-Datasets\show_reviews.tsv", delimiter = '\t', quoting =3,  low_memory=False)

# Cleaning the texts. Run  for i to the # of records available (for example 1000) below,then import re below to start building the corpus
# This model uses text data set (columens separated by tabs), where the first column contains the suscriber text format comments and the 2nd column contains the rating (ei: like/no lke)
## good/bad, like/no like represented as 1 (if favorable) or 0 if not favorable).
## Script should be tested using other datasets (once these are in adequate format)
## The script will run model to issue a prediction for upcoming new record.  New entered record will also help updating bag of words and to enhance predictability of the model)
#Applying stemming
import re
## Use re.sub to note what we don't want to remove
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
# To help you remove irrelevant words to set categories, download nltk, split to have a list of dif. words
## use "set" to work faster, if we run below it will remove "this". 
## corpus =[]below will be the new list of xx amount of cleaned reviews
## review= re.sub... is what is called token pattern found as parameter under CountVectorizer object
# Putting the letters of review with lower cares
# A list of words per record is created with split
review=review.lower()
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[]
for i in range(0, len(dataset.index)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review =review.split()
    ps =PorterStemmer()
    review =[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model (containing up to 1500 words as shown by X in variable explorer; were more if cv() left), applying the sparsity, also can be dealt with dim red.
##   X is the sparse matrix,(to array()is to matrix)
## is the dependent var vector. Index is ilo is 1 as it is where the we have all results of positives or negatives
## Number of words may be modified, no to exceed those available based on the dataset updates/uniqueness, or a function may be written 
## to adjust it
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set (X and y were already set; feature scaling is not neeeded)
# as most are 1/0
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix; assing accuracy of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

##Taking the last record to print the predicion about if suscriber has favorable view/track or not

Modelrows= len(dataset.index)

## Taking the last record of the dataset to report status/ projection
df1= dataset.iloc[-1]

 
def function (df1):
    if df1 ==1:
        return "The suscriber is positive, in good track"
       
    else:
        return "The suscriber is not in a favorable track"
       
