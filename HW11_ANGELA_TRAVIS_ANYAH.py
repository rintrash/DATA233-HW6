#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from textblob import TextBlob as tb
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import sys


# In[2]:


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
        if len(words) > 10000:
            return 1.0
        else:
            return len(words) / 10000
        
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


# In[3]:


scores = []
texts = []
pol = []
sub = []
length = []
read = []
for text in textValues:
    if type(text) != str:
        continue
    score = sentenceScore(text)
    pol.append(score.getPolarity())
    sub.append(score.getSubjectivity())
    length.append(score.getLength())
    read.append(score.getReadability())
    scores.append(score.calculateScore())
    texts.append(text)
    
quality_name = []
#converts scores to categories
for score in scores:
    if float(score) > 0.65:
        quality_name.append("great")
    elif float(score) > 0.55:
        quality_name.append("good")
    else:
        quality_name.append("poor")


# In[4]:


data = {'polarity':pol, 'subjectivity': sub, 'length': length, 'readability': read, 'quality':quality_name}
df = pd.DataFrame(data)


# In[7]:


# create a simple function to tokenize messages into distinct words
from typing import Set
import re

def tokenize(text: str) -> Set[str]:
    text = text.lower()                         # Convert to lowercase,
    all_words = re.findall("[a-z0-9']+", text)  # extract the words, and
    return set(all_words)                       # remove duplicates.

assert tokenize("Data Science is science") == {"data", "science", "is"}



# define a type for our training data
from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool
        
        

# As our classifier needs to keep track of tokens, counts, and labels from the training data, we’ll make it a class.
from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)    # we refer to nonspam emails as ham emails
        self.spam_messages = self.ham_messages = 0

    # Next, we’ll give it a method to train it on a bunch of messages
    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # Increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # Increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1
    
    # Ultimately we’ll want to predict P(spam | token). 
    #As we saw earlier, to apply Bayes’s theorem we need to know P(token | spam) and P(token | ham) 
    #for each token in the vocabulary. So we’ll create a “private” helper function to compute those:
    
    def _probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | spam) and P(token | ham)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham    
    
    # finally we have the predict function
    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Iterate through each word in our vocabulary
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)

            # If *token* appears in the message,
            # add the log probability of seeing it
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # Otherwise add the log probability of _not_ seeing it,
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / ((prob_if_spam + prob_if_ham) + 0.000001)

spam = [df.quality[i] == "poor" for i in range(len(df.quality))]
messages = [Message(texts[i], spam[i]) for i in range(len(spam))]    
    
import random
from typing import TypeVar, List, Tuple

X = TypeVar('X')  # generic type to represent a data point

def split_data(messages: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data2 = messages.copy()                    # Make a shallow copy
    cut = int(len(data2) * prob)       # Use prob to find a cutoff
    return data2[:cut], data2[cut:]     # and split the shuffled list there.

random.seed(0)      # just so you get the same answers as me
train_messages, test_messages = split_data(messages, 0.75)


# In[ ]:





# In[10]:


from collections import Counter

predictions = [(message, model.predict(message.text))
               for message in test_messages]

# Assume that spam_probability > 0.5 corresponds to spam prediction
# and count the combinations of (actual is_spam, predicted is_spam)
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                           for message, spam_probability in predictions)

print(confusion_matrix)


recall = confusion_matrix[(True,True)] / (confusion_matrix[(True,True)] + confusion_matrix[(True,False)])
print(recall)

precision = confusion_matrix[(True,True)] / ((confusion_matrix[(True,True)] + confusion_matrix[(False,True)]) + 0.000001)
print(precision)


# In[8]:


model = NaiveBayesClassifier()
model.train(train_messages)


# In[73]:


obj.predict("Hi, I'd like to sell you stuff.")

