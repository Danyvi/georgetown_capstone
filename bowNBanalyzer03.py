# import necessary tools
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


start_time = time.time()


# load data into DataFrame
comments = pd.read_csv('attack_annotated_comments.tsv', sep='\t', index_col=0)
annotations = pd.read_csv('attack_annotations.tsv', sep='\t')

# print the # of unique rev_id
print('There are', len(annotations['rev_id'].unique()), 'unique rev_id')

# labels a comment as an attack if the majority of annotators did so
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

# insert labels in comments
comments['attack'] = labels

# Parsing: remove newline and tab tokens
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

print('This is your DataFrame:\n')
print(comments.head())
print('These are the head of the comments classified as an attack\n')
print(comments.query('attack')['comment'].head())

# create y which is the outcome label the model has to learn
y = comments['attack']

'''
split the dataframe into training and testing data
split the features (that are the comment column) and the label y (attack) based on a given test size such as 0.33 (33%)
random state, so we can have repeatable results when we run the code again
the function will take 33% of rows to be marked as test data and move them from the training data.
the test data is later used to see how the model has learned

The resulting data from train_test_split() are training data as X_train, training labels as y_train,
testing data as X_test and testing labels as y_test
'''

X_train, X_test, y_train, y_test = train_test_split(comments['comment'], y, test_size=0.33, random_state=53)

# create count vectorizer that turn the text into a bag-of-words vectors
# each tokens acts as a feature for the machine learning classification problem


count_vectorizer = CountVectorizer()

# fit transform on the training data to create a bag-of-words vectors
# it will generate a mapping of words with IDs and vectors representing how many times words appears in the comment

count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)


load_time = time.time()


# building the Naive Bayes classifier
# importing the naive bayes model class MultinomialNB (NaiveBayes), which works well with count_vectorizers as it expects integer inputs

from sklearn.naive_bayes import MultinomialNB

# metric model to evaluate the model performance
from sklearn import metrics

# class inizialization  and fit calling on training data
nb_classifier = MultinomialNB()
nb_classifier.fit(count_train, y_train)

build_time = time.time()

print(nb_classifier)


# predict with the count_vectorizer test data
# predict will use the trained model to predict the label based on the test data vectors
# we save the predicted labels in variable pred to test the accuracy
pred = nb_classifier.predict(count_test)

# testing accuracy
print(metrics.accuracy_score(y_test, pred))

# further evaluation of our model with confusion matrix which shows correct/incorrect labels
print(metrics.confusion_matrix(y_test, pred, labels=[False, True]))
print(metrics.classification_report(y_test, pred))

eval_time = time.time()

print("Times: %0.3f sec loading, %0.3f sec building, %0.3f sec evaluation" % (load_time - start_time, build_time - load_time, eval_time - build_time,))
print("Total time: %0.3f seconds" % (eval_time - start_time))
