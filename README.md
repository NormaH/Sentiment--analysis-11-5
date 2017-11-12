# Sentiment--analysis-11-5
Sentiment analysis from a dataset
Sentiment analysis from a dataset A text file (taking tabs as delimiter as text content will likely contain commas )
with sentiment content (usually where a record is a line) is processed through the model. The present model used Naive Bayes 
prediction algorithm but other algorithm (such as decision trees) may be used. For purpose of timing delivery of a model working
no other models were tested.
You may find text content files at the University of Irvine California datasets to test the model, and to identify if the file you
intent to submit to the model may not face formatting needs before processing. 
Location (content was placed in this folder as gif image of the ongoing code execution showing the variables).
( https://www.dropbox.com/s/41vk97uqlza3wec/NLP-sentiment-analysis-executable%20code%20show%201000th%20record-Amazon%20review%20dataset.GIF?dl=0 ) you may find screen picture of the code running and some of the outputs including the confusion matrix, variable explorer and the last record in the file called in.

Also below are the other than confusion matrix performance metrics of the model(based on a limited sample of 200 tested samples):

Recall for code "0" or no engaged is 0.83  and for code "1" or engaged is 0.56
F1 score for code "0" is 0.69 and for code "1" is 0.66

Considering the number of tested samples, the model performs reasonable; increasing the number of samples available for test the model
should improve.

