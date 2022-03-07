# name 1: Angela You
# name 2: Anyah Arboine
# name 3: Travis McDaneld


from textblob import TextBlob as tb


class sentenceQuality():

    def __init__(self):
        pass

    def calculateScores(self, text):
        # please implement this function
        # input: any tweet text
        # output: a list of scores for the tweet, it must include: score for length, score for Polarity,
        # score for Subjectivity, and at least readability score
        b = tb(text)

        #length
        words = b.split()
        length_score = 0.0
        if len(words) > 20:
            length_score = 1.0
        else:
            length_score = len(words) / 20

        #polarity
        pol = b.sentiment.polarity
        if pol < 0:
            true_pol = 0.5 - (abs(pol) / 2)
        elif pol > 0:
            true_pol = 0.5 + (pol / 2)
        else:
            true_pol = 0.5

        #subjectivity
        subjectivity = b.subjectivity

        #readability
        lines = b.split(".")
        words = b.split(' ')
        characters = len(b.replace(".", " "))
        ARIvalue = 4.71 * (characters / len(words)) + (0.5 * (len(words) / len(lines))) - 21.43
        readability_score = (round(ARIvalue) - 1) / 13

        return [length_score, true_pol, subjectivity,
                readability_score]

    def calculateQuality(self, text):
        # please implement this function to calculate a final quality score between 0 and 1
        # Input: a list of scores, which is the output of calculateScores
        # output: 0 means low quality, 1 mean high quality

        return sum(self.calculateScores(text)) / 4


# # this is for testing only
s = "DATA 233 is a wonderful class! We do projects, like this, where we try to calculate quality"
obj = sentenceQuality()

print("The scores for your input is " + str(obj.calculateScores(s)))

print("The final quality for your input is " + str(obj.calculateQuality(s)))
