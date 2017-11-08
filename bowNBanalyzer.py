# import necessary tools

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# load data into DataFrame
df1 = pd.read_csv('attack_annotated_comments.tsv', sep='\t', header=0)
df2 = pd.read_csv('attack_annotations.tsv', sep='\t', header=0)

df_joined = pd.merge(df1, df2, on='rev_id')

# create y which is the outcome label the model has to learn
y = df_joined['attack']
'''
split the dataframe into training and testing data
split the features (that are the comment column) and the label y (attack) based on a given test size such as 0.33 (33%)
random state, so we can have repeatable results when we run the code again
the function will take 33% of rows to be marked as test data and move them from the training data.
the test data is later used to see how the model has learned

The resulting data from train_test_split() are training data as X_train, training labels as y_train,
testing data as X_test and testing labels as y_test
'''

X_train, X_test, y_train, y_test = train_test_split(df_joined['comment'], y, test_size=0.33, random_state=53)

# create count vectorizer that turn the text into a bag-of-words vectors
# each tokens acts as a feature for the machine learning classification problem

count_vectorizer = CountVectorizer()

# fit transform on the training data to create a bag-of-words vectors
# it will generate a mapping of words with IDs and vectors representing how many times words appears in the comment

count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)

# building the Naive Bayes classifier
# importing the naive bayes model class MultinomialNB (NaiveBayes), which works well with count_vectorizers as it expects integer inputs

from sklearn.naive_bayes import MultinomialNB

# metric model to evaluate the model performance
from sklearn import metrics

# class inizialization  and fit calling on training data
nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)

# predict with the count_vectorizer test data
# predict will use the trained model to predict the label based on the test data vectors
# we save the predicted labels in variable pred to test the accuracy
pred = nb_classifier.predict(count_test)

# testing accuracy
print(metrics.accuracy_score(y_test, pred))

# further evaluation of our model with confusion matrix which shows correct/incorrect labels
print(metrics.confusion_matrix(y_test, pred, labels=[0, 1]))
