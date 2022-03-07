# name 1: Angela You
# name 2: Anyah Arboine
# name 3: Travis McDaneld


from textblob import TextBlob as tb

class sentenceQuality():
    
    def __init__(self, text):
        
        self.b = tb(text)
        
        self.polarity = self.getPolarity()
        self.subjectivity = self.getSubjectivity()
        self.length = self.getLength()
        self.readability = self.getReadability()
        
        self.calculateScores()
        
    def calculateScores(self):
        
        return [self.length, self.polarity, self.subjectivity, self.readability]
    
    def calculateQuality(self):
        
        return sum(self.calculateScores()) / 4
        
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
        
        
    def getReadability(self, tweet):
        lines = tweet.split(".")
        words = tweet.split(' ')
        characters = len(tweet.replace("."," "))
        ARIvalue = 4.71*(characters/len(words)) + (0.5*(len(words)/len(lines))) - 21.43
        return round(ARIvalue)
            

# class sentenceQuality():
#     def __init__(self):
#         # do some initialization, optional
#         pass

#     def calculateScores(self, tweet):
#         # please implement this function
#         # input: any tweet text
#         # output: a list of scores for the tweet, it must include: score for length, score for Polarity, score for Subjectivity, and at least one score of the following:
#         # https://en.wikipedia.org/wiki/Automated_readability_index
#         # https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
#         # https://en.wikipedia.org/wiki/Gunning_fog_index
#         # https://en.wikipedia.org/wiki/SMOG
#         # https://en.wikipedia.org/wiki/Fry_readability_formula
#         # https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
#         # You should implement at least one score

#         return [0.1, 0.2, 0.3, 0.5, 0.6]
#         pass

#     def calculateQuality(self, scores):
#         # please implement this function to calculate a final quality score between 0 and 1
#         # Input: a list of scores, which is the output of calculateScores
#         # output: 0 means low quality, 1 mean high quality

#         return 0.5
#         pass


# # this is for testing only
s = "DATA 233 is a wonderful class!"
obj = sentenceQuality(s)

print("The scores for your input is " + str(obj.calculateScores()))

print("The final quality for your input is " + str(obj.calculateQuality(obj.calculateScores())))
