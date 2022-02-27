

class TwitterPositive():
    def __init__(self):
        # do some initialization, optional
        pass

    def evaluateTweet(self, tweet):
        # please implement this function
        # input: any tweet text
        # output: a score [0,1], 0 means it is low quality and negative, 1 means it is high quality and positive

        neg_words = readFile("negative-words.txt")
        pos_words = readfile("positive-words.txt")
        score = 0.5
        
        for word in tweet: 
            
            if word in pos_words: 
                score += 0.05
            if word in neg_words:
                score -= 0.05
            
        if score > 1.0:
            score = 1.0
        if score < 0.0:
            score = 0.0
        
        return score 

    def readFile(self):
        file = open(fileName, "r")  
        words = fileObj.read().splitlines()  # puts file into array
        file.close()
        return words

# this is for testing only
obj = TwitterPositive()
if obj.evaluateTweet("DATA 233 is a wonderful class!") >= 0.5:
    print("Great, it is positive")
else:
    print("negative")
    
