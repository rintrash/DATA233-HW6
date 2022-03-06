# name 1:
# name 2:
# name 3:


class sentenceQuality():
    def __init__(self):
        # do some initialization, optional
        pass

    def calculateScores(self, tweet):
        # please implement this function
        # input: any tweet text
        # output: a list of scores for the tweet, it must include: score for length, score for Polarity, score for Subjectivity, and at least one score of the following:
        # https://en.wikipedia.org/wiki/Automated_readability_index
        # https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
        # https://en.wikipedia.org/wiki/Gunning_fog_index
        # https://en.wikipedia.org/wiki/SMOG
        # https://en.wikipedia.org/wiki/Fry_readability_formula
        # https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
        # You should implement at least one score

        return [0.1, 0.2, 0.3, 0.5, 0.6]
        pass

    def calculateQuality(self, scores):
        # please implement this function to calculate a final quality score between 0 and 1
        # Input: a list of scores, which is the output of calculateScores
        # output: 0 means low quality, 1 mean high quality

        return 0.5
        pass


# this is for testing only
obj = sentenceQuality()
s = "DATA 233 is a wonderful class!"

print("The scores for your input is " + str(obj.calculateScores(s)))

print("The final quality for your input is " + str(obj.calculateQuality(obj.calculateScores(s))))
