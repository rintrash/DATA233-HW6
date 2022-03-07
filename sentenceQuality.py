# name 1: Angela You
# name 2: Anyah Arboine
# name 3: Travis McDaneld


from textblob import TextBlob as tb

class sentenceQuality():
        
    def calculateScores(self, text):
        
        b = tb(text)
        
        polarity = self.getPolarity(b)
        subjectivity = self.getSubjectivity(b)
        length = self.getLength(b)
        readability = self.getReadability(b)
        
        results = [length, polarity, subjectivity, readability]
        return results
    
    def calculateQuality(self, text):
        
        return sum(text)/4
        
    def getPolarity(self, text):
        
        pol = text.sentiment.polarity
        if pol < 0:
            true_pol = 0.5 - (abs(pol)/2)
        elif pol > 0:
            true_pol = 0.5 + (pol/2)
        else:
            true_pol = 0.5
        return true_pol
    
    def getSubjectivity(self, text):
        
        sub = text.subjectivity
        return sub
    
    def getLength(self, text):
        words = text.split()
        if len(words) > 20:
            return 1.0
        else:
            return len(words) / 20
        
        
    def getReadability(self, text):
        lines = text.split(".")
        words = text.split(' ')
        characters = len(text.replace("."," "))
        ARIvalue = 4.71*(characters/len(words)) + (0.5*(len(words)/len(lines))) - 21.43
        value = round(ARIvalue) - 1
        trueValue = value/13
        if trueValue > 1:
            trueValue = 1
        return trueValue

# # this is for testing only
# obj = sentenceQuality()
# s = "good, great, wonderful, happy"

# print("The scores for your input is " + str(obj.calculateScores(s)))

# print("The final quality for your input is " + str(obj.calculateQuality(obj.calculateScores(s))))
