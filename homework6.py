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
        # Removing punctuations in string
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ele in tweet:
            if ele in punc:
                tweet = tweet.replace(ele, "")
        tweet = tweet.lower()

        tweet = tweet.split()

        neg_words = self.readFile("negative-words.txt")
        pos_words = self.readFile("positive-words.txt")
        emphasis_words = ["very", "extremely", "super", "absolutely", "positively", "completely", "certainly"]

        score = 0.5
        emp = 1.0

        for i in range(len(tweet)):
            if tweet[i] in pos_words:
                score += 0.1 * emp
                emp = 1.0

            if tweet[i] in neg_words:
                score -= 0.1 * emp
                emp = 1.0

            if tweet[i] in emphasis_words:
                emp *= 2

        if score > 1.0:
            score = 1.0
        if score < 0.0:
            score = 0.0

        return score

# this is for testing only
obj = TwitterPositive()
tot_score = obj.evaluateTweet("DATA 233 is very very  bad good!")
if tot_score >= 0.5:
    print(tot_score)
    print("Great, it is positive")
else:
    print(tot_score)
    print("it is negative")
