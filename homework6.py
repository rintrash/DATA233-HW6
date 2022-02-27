

class TwitterPositive():
    def __init__(self):
        # do some initialization, optional
        pass

    def evaluateTweet(self, tweet):
        # please implement this function
        # input: any tweet text
        # output: a score [0,1], 0 means it is low quality and negative, 1 means it is high quality and positive

        return 0.5
        pass


# this is for testing only
obj = TwitterPositive()
if obj.evaluateTweet("DATA 233 is a wonderful class!") >= 0.5:
    print("Great, it is positive")
else:
    print("negative")
    
