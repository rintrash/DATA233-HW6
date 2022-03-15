'''
Group: Anyah Arboine, Travis McDaneld, Angela You
Date: 3/10
Class DATA 233, Dr. Cao
Project: HW9: K Nearest Neighbor
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#we need to extract the data (the sentences) and the target from our csv file

#X = ***list of sentences***
#y = ***list of targets***

#I'm not sure what random_state=42 does but this splits the data into 20% testing, 80% training
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7) #we can change number of neighbors later
knn.fit(X_train, y_train) #fits data 

print(knn.predict(X_test)) #test predictions
print(knn.score(X_test, y_test)) #accuracy of predictions

def raw_majority_vote(): 
  pass
