# georgetown_capstone

I have used as an example the 2 files from wikidetox attack_annotated_comments.tsv and attack_annotations.tsv


I have used bag-of-words vectorization and Naive Bayes model as a classifier.

The Naive Bayes model is commonly used for testing NLP classification problems.
This model try to answer "given a particular piece of data, how likely is a particular outcome?"
Each word acts as a feature from CountVectorizer (helping classifying our text using probability)
The Naive Bayes class MultinomialNB works well with count vectorizers as it expect integer inputs


I have used metric model to evaluate the model performance obtaining 86% and also a confusion matrix to further avaluate the model showing the correct/incorrect labels. The confusion matrix, from the metrics module, takes the test labels, the predictions and a list of labels, in this case [0,1] for non attacks/attacks 
