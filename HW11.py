# name 1: Travis McDaneld
# name 2: Anyah Arboine
# name 3: Angela You


class NBsentenceQuality():
    def __init__(self):
        # do some initialization, optional
        pass
    def trainNB(self, trainingData, NBmodel):
        # traing a NB model on the training dataset, your group should find a training dataset with three different qualities

        pass

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

print("The final quality for your input using NB is " + str(obj.Quality_NB(s)))
