'''
Authors: Manas Gaur, Amanuel Alambo
Instructor: Dr. keke Chen
classification

'''

import codecs
import random
from sklearn import model_selection
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC
from sklearn.model_selection import cross_val_score
import warnings


#method to shuffle training data
def shuffle_train_data(file_path, output_file_path):
    with codecs.open(file_path, mode='r', encoding='utf-8') as f:
        data = f.read()
        # Split on \n
        blocks = data.split('\n')
        # Shuffle splits
        random.shuffle(blocks)

    with codecs.open(output_file_path,  mode='w', encoding='utf-8') as output:
        for block in blocks:
            output.write(block)
            # Add the line break
            output.write('\n')

#method to evaluate each classifier performance on f1-macro, recall-macro and precision-macro
def evaluate_classifier(clf, x, y):
    scores_f1_macro = cross_val_score(clf, x.toarray(), y, cv=5, scoring='f1_macro')
    scores_recall_macro = cross_val_score(clf, x.toarray(), y, cv=5, scoring='recall_macro')
    scores_precision_macro = cross_val_score(clf, x.toarray(), y, cv=5, scoring='precision_macro')

    mean_std_f1 = ("f1-macro : %0.2f (+/- %0.2f)" % (scores_f1_macro.mean(), scores_f1_macro.std() * 2))
    mean_std_precision = ("precision-macro : %0.2f (+/- %0.2f)" % (scores_precision_macro.mean(), scores_precision_macro.std() * 2))
    mean_std_recall = ("recall-macro : %0.2f (+/- %0.2f)" % (scores_recall_macro.mean(), scores_recall_macro.std() * 2))

    return mean_std_f1,mean_std_precision,mean_std_recall


#MultinomialNB classifer
def MultinomialNB_classifer(x,y,x_train,y_train,x_test,y_test):
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    print("Train accuracy:",train_score)
    print("Test accuracy:",test_score)
    #print(evaluate_classifier(clf,x,y))
    return evaluate_classifier(clf,x,y)

#BernoulliNB classifer
def BernoulliNB_classifier(x,y,x_train,y_train,x_test,y_test):
    clf = BernoulliNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    print("Train accuracy:",train_score)
    print("Test accuracy:",test_score)
    #print(evaluate_classifier(clf,x,y))
    return evaluate_classifier(clf,x,y)

#KNeighbors classifer
def KNeighbors_classifier(x,y,x_train,y_train,x_test,y_test):
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    print("Train accuracy:",train_score)
    print("Test accuracy:",test_score)
    #print(evaluate_classifier(clf,x,y))
    return evaluate_classifier(clf,x,y)

#svm_SVC classifer
def svm_SVC_classifer(x,y,x_train,y_train,x_test,y_test):
    clf = SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    print("Train accuracy:",train_score)
    print("Test accuracy:",test_score)
    #print(evaluate_classifier(clf,x,y))
    return evaluate_classifier(clf,x,y)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    shuffle_train_data('sample_training_data_file.TFIDF', 'shuffled_train_data.txt') 
    
       
    feature_vectors, targets = load_svmlight_file("shuffled_train_data.txt")
    print("Dimension of feature vectors:", feature_vectors.shape)
    print("Dimension of target vectors:", targets.shape)

    #x_test and y_test are the held-out dataset
    ## Splitting the whole dataset for training and testing
    x_train, x_test, y_train, y_test = model_selection.train_test_split(feature_vectors, targets, test_size = 0.2, random_state = 41) #75-25 split

    
    #MultinomialNB_classifer
    print("MultinomialNB Results:")
    print(MultinomialNB_classifer(feature_vectors,targets,x_train,y_train, x_test,y_test))

    #Bernoulli NB classifier
    print("BernoulliNB Results:")
    print(BernoulliNB_classifier(feature_vectors,targets,x_train,y_train, x_test,y_test))

    #K Neighbors classifier
    print("KNeighbors Results:")
    print(KNeighbors_classifier(feature_vectors,targets,x_train,y_train, x_test,y_test))
    
    #SVM classifier
    print("svm_SVC Results:")
    print(svm_SVC_classifer(feature_vectors,targets,x_train,y_train, x_test,y_test))
    






