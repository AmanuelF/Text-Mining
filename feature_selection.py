'''
Authors: Manas Gaur, Amanuel Alambo
Instructor: Dr. keke Chen
feature selection

'''

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.datasets import load_svmlight_file
import warnings
from sklearn import model_selection
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import sys
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import BernoulliNB


import scipy.sparse

import classification   #importing program 'classification'---for testing the 4 classifiers

#method to call MultinomialNB classifier in program 'classification' and 
def call_to_MultinomialNB(X_new1, X_new2, y):
	print('Testing classifiers(MultinomialNB) with chi2 selected features')
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X_new1, targets, test_size = 0.2, random_state = 15435)
	f_chi2,_,_ = classification.MultinomialNB_classifer(X_new1,targets,x_train,y_train, x_test,y_test)

	print('Testing classifiers(MultinomialNB) with mutual information selected features')
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X_new2, targets, test_size = 0.2, random_state = 15435)
	f_mi,_,_ = classification.MultinomialNB_classifer(X_new2,targets,x_train,y_train, x_test,y_test)

	return f_chi2,f_mi

def call_to_BernoulliNB(X_new1, X_new2, y):
	print('Testing classifiers(BernoulliNB_classifier) with chi2 selected features')
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X_new1, targets, test_size = 0.2, random_state = 15435)
	f_chi2,_,_ = classification.BernoulliNB_classifier(X_new1,targets,x_train,y_train, x_test,y_test)

	print('Testing classifiers(BernoulliNB_classifier) with mutual information selected features')
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X_new2, targets, test_size = 0.2, random_state = 15435)
	f_mi,_,_ = classification.BernoulliNB_classifier(X_new2,targets,x_train,y_train, x_test,y_test)

	return f_chi2,f_mi

def call_to_KNeighbors(X_new1, X_new2, y):
	print('Testing classifiers(KNeighbors_classifier) with chi2 selected features')
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X_new1, targets, test_size = 0.2, random_state = 15435)
	f_chi2,_,_ = classification.KNeighbors_classifier(X_new1,targets,x_train,y_train, x_test,y_test)

	print('Testing classifiers(KNeighbors_classifier) with mutual information selected features')
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X_new2, targets, test_size = 0.2, random_state = 15435)
	f_mi,_,_ = classification.KNeighbors_classifier(X_new2,targets,x_train,y_train, x_test,y_test)

	return f_chi2,f_mi

def call_to_svm_SVC_classifer(X_new1, X_new2, y):
	print('Testing classifiers(svm_SVC_classifer) with chi2 selected features')
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X_new1, targets, test_size = 0.2, random_state = 15435)
	f_chi2,_,_ = classification.svm_SVC_classifer(X_new1,targets,x_train,y_train, x_test,y_test)

	print('Testing classifiers(svm_SVC_classifer) with mutual information selected features')
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X_new2, targets, test_size = 0.2, random_state = 15435)
	f_mi,_,_ = classification.svm_SVC_classifer(X_new2,targets,x_train,y_train, x_test,y_test)

	return f_chi2,f_mi

#method to plot learning curve given a dataframe and model name
def plot_learning_curve(title, df):
	    
	    plt.figure()
	    plt.title(title)
	    plt.xlabel("K(top features)")
	    plt.ylabel("Score")

	    chi2_f1_mean, chi2_f1_std, mi_f1_mean, mi_f1_std = list(), list(), list(), list()

	    for i,row in df.iterrows():
	    	chi2_f1_mean.append(float(row['chi2_f1'].split(': ')[1].split(' (+/- ')[0]))
	    	chi2_f1_std.append(float(row['chi2_f1'].split(': ')[1].split(' (+/- ')[1].rstrip(')')))

	    	mi_f1_mean.append(float(row['mi_f1'].split(': ')[1].split(' (+/- ')[0]))
	    	mi_f1_std.append(float(row['mi_f1'].split(': ')[1].split(' (+/- ')[1].rstrip(')')))

	    
	    train_sizes = np.array(range(100, (df.shape[0]+1)*100, 100))

	    plt.grid()
	    plt.fill_between(train_sizes, np.array(chi2_f1_mean) - np.array(chi2_f1_std),
	                     np.array(chi2_f1_mean) + np.array(chi2_f1_std), alpha=0.1,
	                     color="r")

	    plt.fill_between(train_sizes, np.array(mi_f1_mean) - np.array(mi_f1_std),
	                     np.array(mi_f1_mean) + np.array(mi_f1_std), alpha=0.1,
	                     color="g")

	    
	    plt.plot(train_sizes, np.array(chi2_f1_mean), 'o-', color="r",
	             label="chi2 f1-score")
	    plt.plot(train_sizes, np.array(mi_f1_mean), 'o-', color="g",
	             label="MI f1-score")

	    plt.legend(loc="best")
	    return plt

if __name__ == '__main__':
	warnings.filterwarnings('ignore')
	feature_vectors, targets = load_svmlight_file("shuffled_train_data.txt")
	X = feature_vectors
	y = targets

	d=list()
	title = ''
	model_name = sys.argv[1]
	if model_name.lower() == 'bnb':
		title = "Learning Curves(BernoulliNB)"
	elif model_name.lower() == 'mnb':
		title = "Learning Curves(MultinomialNB)"
	elif model_name.lower() == 'knb':
		title = "Learning Curves(KNeighbors)"
	elif model_name.lower() == 'svm':
		title = "Learning Curves(SVM)" 
	#title = "Learning Curves",(model_name)

	for i in range(1, 11):
		X_new1 = SelectKBest(chi2, k=i*100).fit_transform(X, y)
		X_new2 = SelectKBest(mutual_info_classif, k=i*100).fit_transform(X, y)
		
		#f_chi2,f_mi = call_to_BernoulliNB(X_new1,X_new2,y)
		#f_chi2,f_mi = call_to_MultinomialNB(X_new1,X_new2,y)
		#f_chi2,f_mi = call_to_KNeighbors(X_new1,X_new2,y)
		if model_name.lower() == 'bnb':
			f_chi2,f_mi = call_to_BernoulliNB(X_new1,X_new2,y)
		elif model_name.lower() == 'mnb':
			f_chi2,f_mi = call_to_MultinomialNB(X_new1,X_new2,y)
		elif model_name.lower() == 'knb':
			f_chi2,f_mi = call_to_KNeighbors(X_new1,X_new2,y)
		elif model_name.lower() == 'svm':
			f_chi2,f_mi = call_to_svm_SVC_classifer(X_new1,X_new2,y)
		else:
			print("Use either MNB(for MultinomialNB), BNB(for BernoulliNB, KNB(for KNeighbors), or SVM")

		d.append({'k':i*100, 'chi2_f1': f_chi2, 'mi_f1': f_mi})
	df = pd.DataFrame(d)	


	#title = "Learning Curves (BernoulliNB)"
	#title = "Learning Curves (MultinomialNB)"
	#title = "Learning Curves (KNeighbors)"
	plot_learning_curve(title, df)   #call to plotting learning curve method

	plt.show()

