#Aurel Rexhaj
#USING NATURAL LANGUAGE PROCESSING with bag of words

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #quoting 3 igonres double quotes

# Cleaning the texts
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

corpus = []
for i in range(0,1000): 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #removing all but letters
    review = review.lower()
    review = review.split()


    #Stemming -> taking the root of the word
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #use set() in stopwords for faster results on big files
    review = ' '.join(review)
    corpus.append(review)

#create a sparse matrix to remove unnecessary words and create bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #take only 1500 features 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#fitting the model
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
Looking at confusion matrix we see that we get an accuracy of 84% which is ok for the given dataset.
No Dimensionality reduction was performed on the dataset

"""