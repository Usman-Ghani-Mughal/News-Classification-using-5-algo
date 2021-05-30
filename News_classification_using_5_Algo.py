#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[49]:


import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import re
from bs4 import BeautifulSoup
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Data Sets

# In[50]:


df = pd.read_json('News_Category_Dataset_v2.json/News_Category_Dataset_v2.json', lines=True)
df.head()


# In[56]:


df[df['authors']== ''] = 'Patrick Cockburn'


# # Data Prepration

# In[61]:


df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)


# In[62]:


df['news']= [df.loc[x, 'headline']+ ". " + df.loc[x,'short_description'] +" "+ df.loc[x,'authors'] for x in range(len(df))]


# In[63]:


df = df[pd.notnull(df['news'])]


# In[64]:


plt.figure(figsize=(15,9))
df.category.value_counts().plot(kind='bar');


# In[65]:


df.head()


# In[66]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


# In[67]:


def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


# In[68]:


df['news'] = df['news'].apply(clean_text)


# In[69]:


df.head()


# # Extract Y and X from Repared Data Set

# In[70]:


X = df.loc[:,'news']
Y = df.iloc[:,0]
classes = df['category'].unique()


# # Train Test Split

# In[71]:


train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state = 42)


# # Naive Bayes Classifier for Multinomial Models

# In[72]:


nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(train_x, train_y)


# In[73]:


y_pred = nb.predict(test_x)


# In[75]:


print('accuracy %s' % accuracy_score(y_pred, test_y))
print(classification_report(test_y, y_pred,target_names=classes))


# # DecisionTree

# In[76]:


nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', DecisionTreeClassifier(random_state=0)),
              ])
nb.fit(train_x, train_y)


# In[77]:


y_pred = nb.predict(test_x)


# In[78]:


print('accuracy %s' % accuracy_score(y_pred, test_y))
print(classification_report(test_y, y_pred,target_names=classes))


# # Random Forest

# In[79]:


svm = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', RandomForestClassifier(n_estimators=100)),
               ])
svm.fit(train_x, train_y)


# In[80]:


y_pred = svm.predict(test_x)


# In[81]:


print('accuracy %s' % accuracy_score(y_pred, test_y))
print(classification_report(test_y, y_pred,target_names=classes))


# #  Support Vector Machine

# In[82]:


sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(train_x, train_y)


# In[83]:


y_pred = sgd.predict(test_x)


# In[84]:


print('accuracy %s' % accuracy_score(y_pred, test_y))
print(classification_report(test_y, y_pred,target_names=classes))


# # Logistic Regression

# In[88]:


logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
logreg.fit(train_x, train_y)


# In[86]:


y_pred = logreg.predict(test_x)


# In[87]:


print('accuracy %s' % accuracy_score(y_pred, test_y))
print(classification_report(test_y, y_pred,target_names=classes))


# In[ ]:




