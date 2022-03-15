'''
Group: Anyah Arboine, Travis McDaneld, Angela You
Date: 3/10
Class DATA 233, Dr. Cao
Project: HW9: K Nearest Neighbor
'''

### Not using now, saving for later

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split

# #we need to extract the data (the sentences) and the target from our csv file

# #X = ***list of sentences***
# #y = ***list of targets***

# #I'm not sure what random_state=42 does but this splits the data into 20% testing, 80% training
# X_train, X_test, y_train, y_test = train_test_split(
#              X, y, test_size = 0.2, random_state=42)

# knn = KNeighborsClassifier(n_neighbors=7) #we can change number of neighbors later
# knn.fit(X_train, y_train) #fits data 

# print(knn.predict(X_test)) #test predictions
# print(knn.score(X_test, y_test)) #accuracy of predictions

# def raw_majority_vote(): 
#   pass




### Not using this now, saving for later

# class textModel():
    
#     def __init__(self, k, df):
        
#         # X is all data without scores, y is just scores
#         self.X = df.drop('scores', axis=1)
#         self.y = df['scores'].values
        
#         # Splits into training/testing data
#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state=42, stratify=self.y)
#         self.knn = KNeighborsClassifier(n_neighbors=k)
        
#         # Fits data into classifier
#         self.knn.fit(self.X_train, self.y_train)
        
#         self.assess()
        
# #         new_prediction = knn.predict(X_new)
# #         print("Prediction: {}".format(new_prediction))
    
#     def assess(self):
        
#         print(knn.score(X_test, y_test))
    
#     def predict(self):
#         pass
# #         self.y_pred = knn.predict(X)


import math
import statistics
import pandas as pd
from textblob import TextBlob as tb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv('kaggle_data.csv')
texts = df['external_text']
textValues = df['external_text'].values


class sentenceScore():
    
    def __init__(self, text):
        
        self.b = tb(text)
        
        self.polarity = self.getPolarity()
        self.subjectivity = self.getSubjectivity()
        self.length = self.getLength()
        self.readability = self.getReadability()
        
        self.result = self.calculateScore()
        
    def __repr__(self):
        
        return repr(self.result)
        
    def calculateScore(self):
        
        results = (self.polarity + self.subjectivity + self.length + self.readability)/4
        return float(results)
    
    def getPolarity(self):
        
        self.pol = self.b.sentiment.polarity
        if self.pol < 0:
            true_pol = 0.5 - (abs(self.pol)/2)
        elif self.pol > 0:
            true_pol = 0.5 + (self.pol/2)
        else:
            true_pol = 0.5
        return true_pol
    
    def getSubjectivity(self):
        
        self.sub = self.b.subjectivity
        return self.sub
    
    def getLength(self):
        
        words = self.b.split()
        if len(words) > 20:
            return 1.0
        else:
            return len(words) / 20
        
    def getReadability(self):
        
        lines = text.split(".")
        words = text.split(' ')
        characters = len(text.replace("."," "))
        ARIvalue = 4.71*(characters/len(words)) + (0.5*(len(words)/len(lines))) - 21.43
        value = round(ARIvalue) - 1
        trueValue = value/13
        if trueValue > 1:
            trueValue = 1
        return trueValue


scores = []
for text in textValues:
    if type(text) == float:
        text = ""
        scores.append(sentenceScore(text))
    else:
        scores.append(sentenceScore(text))
        
        
textValues = list(textValues)


data = {'texts':textValues, 'scores':scores}
df = pd.DataFrame(data)









import random
from typing import TypeVar, List, Tuple

X = TypeVar('X') # This is a generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]

data = [n for n in range(1000)]               # List of values ranging from 0 to 1000
train, test = split_data(data, 0.75)


# Making sure the function does what we intend
assert len(train) == 750
assert len(test) == 250

assert sorted(train + test) == data

#####################################
#    Pair input/output variables    #
#####################################

Y = TypeVar('Y') # Represents output variables


def train_test_split(xs: List[X], 
                    ys: List[Y], 
                    test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    #Generate indices to be split
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test-pct)
    
    return([xs[i] for i in train_idxs], 
           [xs[i] for i in test_idxs],
           [ys[i] for i in train_idxs],
           [ys[i] for i in test_idxs],)
    pass

##########################################
#    Making sure the code works right    #
##########################################

xs = [x for x in range(1000)]
ys = [2 * x for x in xs]
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)


# Make sure the datasets are the right lengths
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250


# Check that data points are paired correctly
assert all(y == 2 * x for x, y in zip(x_train, y_train))
assert all(y == 2 * x for x, y in zip(x_test, y_test))

#################################
#    Machine learning basics    #
#################################

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    
    pass

assert accuracy(70, 4930, 13930, 981070) == 0.98114

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
    
    pass

assert precision(70, 4930, 13930, 981070) == 0.014

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    
    pass
           
assert recall(70, 4930, 13930, 981070) == 0.005
