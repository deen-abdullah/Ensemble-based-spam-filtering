# ###############################################################################################################################
#                        Course Project
# Coudse Code                   : CPSC 5310
# Author                        : Deen Md Abdullah
# Course Teacher                : Dr. Wendy Osborn
# Project Name                  : Spam Email Filtering
# Training and Testing Dataset  : Enron's Dataset
# Output                        : Generate confusion matrix of three base machine learning model and the proposed ensembled model
# ###############################################################################################################################

import os
import numpy as np
from sklearn.svm import LinearSVC                         # sklearn library
from sklearn.naive_bayes import MultinomialNB             # sklearn library
from sklearn import tree                                  # sklearn library
from collections import Counter           
from sklearn.model_selection import train_test_split      # sklearn library
from tabulate import tabulate                             # To print the confusion matrix as tabular format


###########################################
# method name       : Create_Dictionary
# Input             : Reads all the emails from the dataset
#                     Then collects all the words which are common by removing the less common words and numbers
# Output            : returns 3000 words of dictionary
###########################################
def create_Dictionary(repository_dir):
    emails_dirs = [os.path.join(repository_dir,f) for f in os.listdir(repository_dir)]    
    all_words = []       
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d,f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:
                    for line in m:
                        words = line.split()
                        all_words += words
    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    #dictionary = dictionary.most_common(300)
    
    np.save('dictEnron.npy',dictionary) 
    #print dictionary
    return dictionary


###########################################
# method name       : extract_features
# Input             : Reads all the emails from the dataset and
#                     3000 words dictionary
#                     generates features matrix and labels the emails for training
# Output            : returns features_matrix and train_labels
###########################################
def extract_features(repository_dir): 
    emails_dirs = [os.path.join(repository_dir,f) for f in os.listdir(repository_dir)]  
    docID = 0
    features_matrix = np.zeros((33716,3000))
    train_labels = np.zeros(33716)
    #features_matrix = np.zeros((3371,300))
    #train_labels = np.zeros(3371)
    
    for emails_dir in emails_dirs:
        dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
        for d in dirs:
            emails = [os.path.join(d,f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail) as m:
                    all_words = []
                    for line in m:
                        words = line.split()
                        all_words += words
                    for word in all_words:
                      wordID = 0
                      for i,d in enumerate(dictionary):
                        if d[0] == word:
                          wordID = i
                          features_matrix[docID,wordID] = all_words.count(word)
                train_labels[docID] = int(mail.split(".")[-2] == 'spam')
                docID = docID + 1                
    return features_matrix,train_labels
    
##############################################################################
# Main Method
# Program executes from here
##############################################################################
repository_dir = 'EnronDataSet'
#repository_dir = 'SampleData'

dictionary = create_Dictionary(repository_dir)
features_matrix,labels = extract_features(repository_dir)

np.save('enronFeaturesMatrix.npy',features_matrix)
np.save('enronLabels.npy',labels)

print "Feature Matrix Size: ",features_matrix.shape
print "Labels Size: ",labels.shape
print "Total Ham emails: ", sum(labels==0), "and Total Spam emails: ",sum(labels==1)

# Random selection 60% emails for training and 40% emails for testing when test_size=0.40                                               
X_train, X_test, y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40) 

model1 = LinearSVC(max_iter=1000000)   # SVM model
model2 = MultinomialNB()               # NB model
model3 = tree.DecisionTreeClassifier() # DT Model

model1.fit(X_train,y_train)            # Train SVM model
model2.fit(X_train,y_train)            # Train NB model
model3.fit(X_train,y_train)            # Train DT model

#### Prediction Result ####
result1 = model1.predict(X_test)       # Test SVM model
result2 = model2.predict(X_test)       # Test NB Model
result3 = model3.predict(X_test)       # Test DT Model


###############################################################
# My contribution: Ensamble three models to get majority result
###############################################################
result4 = []
i=0
for r in result1:
    t = result1[i]+ result2[i]+ result3[i]
    if t>=2.0:
        result4.append(1.0)
    else:
        result4.append(0.0)
    i+=1
################################################################


#################
##  Evaluation  #
#################

##### Evaluation with Confusion Matrix #####
print(" ")
print("SVM (Confusion Matrix):")

true_ham=0
false_ham=0
true_spam=0
false_spam=0
i=0

for r in result1:
    if(y_test[i]==0 and result1[i]==0):
        true_ham+=1;
    elif (y_test[i]==0 and result1[i]==1):
        false_ham+=1
    elif (y_test[i]==1 and result1[i]==0):
        false_spam+=1
    else:
        true_spam+=1
    i+=1

print tabulate([['Ham', true_ham, false_ham], ['Spam', false_spam,true_spam]], headers=['SVM', 'HAM', 'SPAM'])


##############################################
print(" ")
print("Naive Bayes (Confusion Matrix):")

true_ham=0
false_ham=0
true_spam=0
false_spam=0
i=0

for r in result2:
    if(y_test[i]==0 and result2[i]==0):
        true_ham+=1;
    elif (y_test[i]==0 and result2[i]==1):
        false_ham+=1
    elif (y_test[i]==1 and result2[i]==0):
        false_spam+=1
    else:
        true_spam+=1
    i+=1

print tabulate([['Ham', true_ham, false_ham], ['Spam', false_spam,true_spam]], headers=['NB', 'HAM', 'SPAM'])


#################################################
print(" ")
print("Decision Tree (Confusion Matrix):")

true_ham=0
false_ham=0
true_spam=0
false_spam=0
i=0

for r in result3:
    if(y_test[i]==0 and result3[i]==0):
        true_ham+=1;
    elif (y_test[i]==0 and result3[i]==1):
        false_ham+=1
    elif (y_test[i]==1 and result3[i]==0):
        false_spam+=1
    else:
        true_spam+=1
    i+=1

print tabulate([['Ham', true_ham, false_ham], ['Spam', false_spam,true_spam]], headers=['DT', 'HAM', 'SPAM'])


##############################################
print(" ")
print("Ensemble (Confusion Matrix):")

true_ham=0
false_ham=0
true_spam=0
false_spam=0
i=0

for r in result4:
    if(y_test[i]==0 and result4[i]==0):
        true_ham+=1;
    elif (y_test[i]==0 and result4[i]==1):
        false_ham+=1
    elif (y_test[i]==1 and result4[i]==0):
        false_spam+=1
    else:
        true_spam+=1
    i+=1

print tabulate([['Ham', true_ham, false_ham], ['Spam', false_spam,true_spam]], headers=['Ensemble', 'HAM', 'SPAM'])


#############
# Thank You #
#############
