# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:08:32 2020

@author: USER
"""


import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

reviews_train = []
for line in open('./movie_data/full_train.txt', 'r', encoding = "utf-8"):
    
    reviews_train.append(line.strip())
    
reviews_test = []
for line in open('./movie_data/full_test.txt', 'r', encoding = "utf-8"):
    
    reviews_test.append(line.strip())
    
target = [1 if i < 12500 else 0 for i in range(25000)]



REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

def preprocess_reviews(reviews):
    
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)



baseline_vectorizer = CountVectorizer(binary=True)
baseline_vectorizer.fit(reviews_train_clean)
X_baseline = baseline_vectorizer.transform(reviews_train_clean)
X_test_baseline = baseline_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X_baseline, target, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))
    
    
    
    
final_model = LogisticRegression(C=0.05)
final_model.fit(X_baseline, target)
print ("Final Accuracy: %s" 
       % accuracy_score(target, final_model.predict(X_test_baseline)))