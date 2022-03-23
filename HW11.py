# name 1: Travis McDaneld
# name 2: Anyah Arboine
# name 3: Angela You


# create a simple function to tokenize messages into distinct words
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

class NBsentenceQuality():
    def __init__(self):
        # do some initialization, optional
        pass
    def trainNB(self):
        # traing a NB model on the training dataset, your group should find a training dataset with three different qualities

        self.readCSV('kaggle_data.csv')

    def readCSV(self, csv):
        df = pd.read_csv(csv)
        #texts = df['external_text']        #do we need this?
        texts = df['external_text'].values  #numpy representation of texts
        targets = df['target'].values       #numpy representation of targets

        target_values = []                   #the good, ok, bad representation of targets
        for target in targets:
            if target > -0.4:
                target_values.append(1)
            elif target > -1.5:
                target_values.append(0)
            else:
                target_values.append(-1)

        #DEBUGGING CODE
        #print("texts: " + texts[:2])
        #print("targets:", target_values[:2])
        return texts, target_values

    def Quality_NB(self, sentence, NBmodel):
        # please implement this function to classify the sentence into three different classes: high, low, and medium quality
        # Input: sentence
        # output: -1 means low quality, 0 means medium quality, 1 means high quality
        # notes: you can reuse the code from the class about NB, and you can add more functions in this class as needed

        return 0
        pass

# this is for testing only
obj = NBsentenceQuality()
s = "DATA 233 is a wonderful class!"
obj.trainNB()
#print("The final quality for your input using NB is " + str(obj.Quality_NB(s)))
