'''
Authors: Manas Gaur, Amanuel Alambo
Instructor: Dr. keke Chen
clustering

'''

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import sys

def plot_learning_curve(title, df):
	    
	    plt.figure()
	    plt.title(title)
	    #if ylim is not None:
	     #   plt.ylim(*ylim)
	    plt.xlabel("K(number of clusters)")
	    plt.ylabel("Score")
	    #train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	    #scores_f1_macro = cross_val_score(clf, x.toarray(), y, cv=5, scoring='f1_macro')
	    #train_scores = f1-macro : 0.07 (+/- 0.00)

	    
	    
	    #train_sizes = np.array(range((2, df.shape[0]+2)))
	    train_sizes = np.array(range(1, (df.shape[0]+1)))
	    #print(train_sizes.shape)
	    #print(len(chi2_f1_mean))
	    #raise KeyboardInterrupt

	    plt.grid()
	   
	    plt.plot(train_sizes, df.silhoutte_score.values, 'o-', color="r",
	             label="silhouette Score")
	    plt.plot(train_sizes, df.nmi_score.values, 'o-', color="g",
	             label="NMI Score")

	    plt.legend(loc="best")
	    return plt

feature_vectors, targets = load_svmlight_file("shuffled_train_data.txt")
#kmeans_model = KMeans(n_clusters=20).fit(feature_vectors)
#single_linkage_model = AgglomerativeClustering(n_clusters=20, linkage='ward').fit(feature_vectors.toarray())

X = feature_vectors
classification_labels = targets

model_name = sys.argv[1]

d=list()
for i in range(2, 25):
	print('number of clusters: ',i)
	if model_name.lower() == 'kmeans':
		kmeans_model = KMeans(n_clusters=i).fit(feature_vectors)
		clustering_labels = kmeans_model.labels_
	elif model_name.lower() == 'agglomerative':
		single_linkage_model = AgglomerativeClustering(n_clusters=i, linkage='ward').fit(feature_vectors.toarray())
		clustering_labels = single_linkage_model.labels_
	else:
		print('Use KMeans or Agglomerative as your argument')

	X = feature_vectors
	classification_labels = targets

	silhoutte_score = metrics.silhouette_score(X, clustering_labels, metric='euclidean')
	nmi_score = metrics.normalized_mutual_info_score(classification_labels, clustering_labels)
	print('silhoutte_score: ',silhoutte_score)
	print('nmi_score: ',nmi_score)
	#X_new1 = SelectKBest(chi2, k=i+1).fit_transform(X, y)
	#X_new2 = SelectKBest(mutual_info_classif, k=i*100).fit_transform(X, y)
	#calls to each classifier for the new features
	#f_chi2,f_mi = call_to_MultinomialNB(X_new1,X_new2,y)
	#f_chi2,f_mi = call_to_BernoulliNB(X_new1,X_new2,y)
	d.append({'k':i, 'silhoutte_score': silhoutte_score, 'nmi_score': nmi_score})
	#call_to_KNeighbors(X_new1,X_new2,y)
	#call_to_svm_SVC_classifer(X_new1,X_new2,y)
df = pd.DataFrame(d)

title = "Document Clustering Evaluation"
plot_learning_curve(title, df)

plt.show()

