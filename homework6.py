class TwitterPositive():
    def __init__(self):
        # do some initialization, optional
        pass

    def readFile(self, fileName):
        file = open(fileName, "r")
        words = file.read().splitlines()  # puts file into array
        file.close()
        return words

    def evaluateTweet(self, tweet):
        # please implement this function
        # input: any tweet text
        # output: a score [0,1], 0 means it is low quality and negative, 1 means it is high quality and positive
        tweet = tweet.split()
        neg_words = self.readFile("negative-words.txt")
        pos_words = self.readFile("positive-words.txt")
        emphasis = ["very", "extremely", "super", "absolutely", "positively", "completely", "certainly"]
        #we could add best, worst, fastest
        score = 0.5
        pos = 0
        neg = 0
        prev = tweet[0]

        for i in range(1, len(tweet)):
            if tweet[i] in pos_words:
                score += 0.05
                pos += 1
            if tweet[i] in neg_words:
                score -= 0.05
                neg += 1

        if score > 1.0 or neg == 0:
            score = 1.0
        if score < 0.0 or pos == 0:
            score = 0.0

        return score

# this is for testing only
obj = TwitterPositive()
score = obj.evaluateTweet("DATA 233 is a wonderful class!")
if score >= 0.5:
    print(score)
    print("Great, it is positive")
else:
    print("negative")
