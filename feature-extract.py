'''
Authors: Manas Gaur, Amanuel Alambo
Instructor: Dr. keke Chen
feature extractor

'''

import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer 
import math
from collections import deque
import os
import json
import subprocess
import sys

nltk.download('stopwords')
nltk.download('punkt')


#class Index used for indexing the entire newsgroup datasets
#source: https://nlpforhackers.io/building-a-simple-inverted-index-using-nltk/
class Index:
    """ Inverted index data structure """
 
    def __init__(self, tokenizer, stemmer=None, stopwords=None):
        """
        tokenizer   -- NLTK compatible tokenizer function
        stemmer     -- NLTK compatible stemmer 
        stopwords   -- list of ignored words
        """
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}
        self.__unique_id = 0    #can be used as docid
        if not stopwords:
            self.stopwords = list()
        else:
            self.stopwords = list(stopwords)
 
    def lookup(self, word):
        """
        Looks up a word in the index
        """
        word = word.lower()
        if self.stemmer:
            word = self.stemmer.stem(word)
 
        return [self.documents.get(id, None) for id in self.index.get(word)]
 
    def add(self, document):
        """
        Add a document string to the index
        """
        #words=[word.lower() for word in words if word.isalpha()]   #added on 0415
        for token in [t.lower() for t in nltk.word_tokenize(document)]:
            if not token.isalpha():
                continue

            if token in self.stopwords:
                continue
 
            if self.stemmer:
                token = self.stemmer.stem(token)
 
            if self.__unique_id not in self.index[token]:
                self.index[token].append(self.__unique_id)
 
        self.documents[self.__unique_id] = document
        self.__unique_id += 1
        
    #method to return inverted index of the entire newsgroup datasets(i.e., term to list of doc-ids mappings)
    def indexed_docs(self): 
        return self.index 
        #return self.documents

    #method to return documents index(i.e., mappings of doc-id(0-1999) and entire document content(subject+body))
    def indexed_docs_I(self): 
        #return self.index 
        return self.documents
 
doc_subdir = {}    #dictionary of doc-id(0-1999) as key and the directory(subdirectory) the doc belongs to

#method to take the root directory path as parameter and parse each doc in each newsgroup subdirectory
def subject_body_index(rootdir):  
    #rootdir = 'mini_newsgroups'
    nDocs = 0  #to keep track of number of indexed docs

    #doc_subdir = {}    #added on 0416----dictionary of doc-id(0-1999) as key and the directory(subdirectory) the doc belongs to
    sub_dir_count = 0  #added 0416----to get over counting the root directory itself----to be used for libsvm 
    doc_count = 0   #increments with every doc read from all subdirectories(there are 2000 docs in total)
    for subdir, dirs, files in os.walk(rootdir):  #iteration through directory 'mini_newsgroup'
        if sub_dir_count > 0:   #In the first round of iteration, the mini_newsgroup directory itself get read---this is to avoid that
            for file in files:
                nDocs += 1
                #cf = open('mini_newsgroups/rec.autos/103806')
                cf = open(os.path.join(subdir, file), encoding='iso-8859-1')
                #print(os.path.join(subdir, file))
                subject = ''
                body = ''

                doc = ''
                for line in cf:
                    try:
                        if 'Lines' in line:
                            n = int(line.strip().split(': ')[1])
                            #print(type(n))
                            body = deque(cf, maxlen=n)
                            body = ''.join(body)
                            #print(body)
                            #index.add(body)

                        if 'Subject' in line:
                            subject = line.strip().split(': ', 1)[1]
                            #print(subject)
                            #index.add(doc_subject)
                    except:
                        continue
                    

                doc = subject + ' @' + body   #a combination of 'subject' and 'body' make up the document to index separated by character '@'
                index.add(doc.strip())
                doc_subdir[doc_count] = subdir.split('/')[1]   #added on 0416---doc_count is in effect docid
                doc_count += 1
        sub_dir_count += 1

    #print(doc_subdir)

#method to generate feature definition file
def feature_defn_gen(feature_defn_file):
    feature_id = 0
    with open(feature_defn_file, 'w') as f:
        for k,v in index.indexed_docs().items():
            feature_id_term_map = (feature_id,k)
            f.write(str(feature_id_term_map))
            f.write('\n')
            feature_id += 1    

#hard-coded class labels
class_1 =  '(comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x)'
class_2 = '(rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey)'
class_3 = '(sci.crypt, sci.electronics, sci.med, sci.space)'
class_4 = '(misc.forsale)'
class_5 = '(talk.politics.misc, talk.politics.guns, talk.politics.mideast)'
class_6 = '(talk.religion.misc, alt.atheism, soc.religion.christian)'
class_definition = defaultdict(list)

#method to generate class definition file
def class_defn_gen(class_defn_file, rootdir):  
    #####################################################################
    #class definition file generation snippet
    sub_dir_count = 0
    with open(class_defn_file, 'w') as f:
        for subdir, dirs, files in os.walk(rootdir):
            if sub_dir_count > 0:      
                if 'comp' in subdir.split('/')[1]:
                        tuple_map = subdir.split('/')[1],class_1
                        f.write(str(tuple_map))
                        f.write('\n')
                elif 'rec' in subdir.split('/')[1]:
                        tuple_map = subdir.split('/')[1],class_2
                        f.write(str(tuple_map))
                        f.write('\n')
                elif 'sci' in subdir.split('/')[1]:
                        tuple_map = subdir.split('/')[1],class_3
                        f.write(str(tuple_map))
                        f.write('\n')
                elif 'misc.forsale' in subdir.split('/')[1]:
                        tuple_map = subdir.split('/')[1],class_4
                        f.write(str(tuple_map))
                        f.write('\n')
                elif 'talk.politics' in subdir.split('/')[1]:
                        tuple_map = subdir.split('/')[1],class_5
                        f.write(str(tuple_map))
                        f.write('\n')
                else:
                    
                        tuple_map = subdir.split('/')[1],class_6
                        f.write(str(tuple_map))
                        f.write('\n')              
                    
            sub_dir_count += 1



def feature_id_gen(feature_defn_file):
    featureids_terms = [line.rstrip('\n') for line in open(feature_defn_file)]
    return featureids_terms
def class_defn_pair_gen(class_defn_file):
    class_definition_file = [line.rstrip('\n') for line in open(class_defn_file)]  #added on 0416
    return class_definition_file

#featureids_terms = [line.rstrip('\n') for line in open('feature_definition_file')]
#class_definition_file = [line.rstrip('\n') for line in open('class_definition_file.csv')]  #added on 0416
class_definition_pairs = {}

#method to compute normalized term frequency of a term in a document
def tf_compute(doc, term):
    tf = round((doc.count(term))/float(len(doc.split(' '))),2)
    return tf

#method to generate TF based training dataset
def training_data_tf_gen(training_data_tf_based,featureids_terms,class_definition_file):    
    #training data file generation snippet---tf-based(term frequency based training dataset)

    #iterates through each newgroup to class mapping
    for pair in class_definition_file:
        key = pair.lstrip('(').rstrip(')').split(', ',1)[0].replace("'",'')
        value = pair.lstrip('(').rstrip(')').split(', ',1)[1].replace("'",'')
        class_definition_pairs[key] = value

    #print(class_definition_pairs)
    count = 0
    doc_list = defaultdict(list)

    with open(training_data_tf_based, 'w') as f:
        for k,v in index.indexed_docs_I().items():
            #print(v)
            news_group = doc_subdir[k]
            class_label = class_definition_pairs[news_group]   #extracts class label given newsgroup name

            if class_label == class_1:
                class_label = 0
            elif class_label == class_2:
                class_label = 1
            elif class_label == class_3:
                class_label = 2
            elif class_label == class_4:
                class_label = 3
            elif class_label == class_5:
                class_label = 4
            elif class_label == class_6:
                class_label = 5

            f.write(str(class_label))
            f.write(' ')  
            for feat_term in featureids_terms:
                #print(feat_term)
                feat_term = feat_term.replace('(','').replace(')','')
                feat,term = feat_term.split(', ')[0],feat_term.split(', ')[1]
                term = term.replace("'", '')
                if term in v:
                    #tf = round((v.count(term))/float(len(v.split(' '))),4)    #normalized term frequency of a term in a document
                    tf = tf_compute(v,term)    #normalized term frequency of a term in a document

                    feat_tf_map = str(feat)+':'+str(tf)

                    f.write(feat_tf_map)
                    f.write(' ')
                    doc_list[class_label].append(feat_tf_map)
                    #doc_list[k].append(feat_tf_map)
            f.write('\n')
    
#method to return the number of files(documents) in a directory---used in function 'nDocs_in_subdir'
def filecount(dir_name):
    # return the number of files in directory dir_name
    dir_name = 'mini_newsgroups/'+str(dir_name)  #subdirectory pathname from current directory
    try:
        return len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
    except:
        return None

#method to return the total number of documents a given document belongs to
def nDocs_in_subdir(docid):
    sub_dir = doc_subdir[docid]
    file_count = filecount(sub_dir)
    return file_count

#method to compute idf score of a term in a document
def idf_compute(nDocs, nDocs_term):
        ''' computes the inverted document frequency for a given term'''
        try:
            idf_score = math.log(nDocs, nDocs_term)
        except:
            idf_score = 1.0
        return round(idf_score,2)


#method to generate IDF based training dataset
def training_data_idf_gen(training_data_idf_based,inv_index,featureids_terms,class_definition_file):
    for pair in class_definition_file:
        key = pair.lstrip('(').rstrip(')').split(', ',1)[0].replace("'",'')
        value = pair.lstrip('(').rstrip(')').split(', ',1)[1].replace("'",'')
        class_definition_pairs[key] = value

    #print(class_definition_pairs)
    count = 0
    doc_list = defaultdict(list)

    #open output file for IDF based training data
    with open(training_data_idf_based, 'w') as f:
        for k,v in index.indexed_docs_I().items():
            #print(v)
            news_group = doc_subdir[k]
            class_label = class_definition_pairs[news_group]

            if class_label == class_1:
                class_label = 0
            elif class_label == class_2:
                class_label = 1
            elif class_label == class_3:
                class_label = 2
            elif class_label == class_4:
                class_label = 3
            elif class_label == class_5:
                class_label = 4
            elif class_label == class_6:
                class_label = 5

            f.write(str(class_label))
            f.write(' ')  
            for feat_term in featureids_terms:
                #print(feat_term)
                feat_term = feat_term.replace('(','').replace(')','')
                feat,term = feat_term.split(', ')[0],feat_term.split(', ')[1]
                term = term.replace("'", '')
                if term in v:
                    nDocs_term = len(inv_index[term])
                    
                    nDocs = nDocs_in_subdir(k)   #number of docs(files) in a subdirectory a document belongs to
                    idf_score = idf_compute(nDocs, nDocs_term)   #100 is number of documents in a single newsgroup
                    
                    feat_idf_map = str(feat)+':'+str(idf_score)

                    #feat_tf_map = str(feat)+':'+str(tf)

                    f.write(feat_idf_map)
                    f.write(' ')
                    doc_list[class_label].append(feat_idf_map)
                    #doc_list[k].append(feat_tf_map)
            f.write('\n')
    

#method to generate TF-IDF based training dataset
def training_data_tf_idf_gen(training_data_tf_idf_based,inv_index,featureids_terms,class_definition_file):
    for pair in class_definition_file:
        key = pair.lstrip('(').rstrip(')').split(', ',1)[0].replace("'",'')
        value = pair.lstrip('(').rstrip(')').split(', ',1)[1].replace("'",'')
        class_definition_pairs[key] = value

    #print(class_definition_pairs)
    count = 0
    doc_list = defaultdict(list)

    with open(training_data_tf_idf_based, 'w') as f:
        for k,v in index.indexed_docs_I().items():
            #print(v)
            news_group = doc_subdir[k]
            class_label = class_definition_pairs[news_group]

            if class_label == class_1:
                class_label = 0
            elif class_label == class_2:
                class_label = 1
            elif class_label == class_3:
                class_label = 2
            elif class_label == class_4:
                class_label = 3
            elif class_label == class_5:
                class_label = 4
            elif class_label == class_6:
                class_label = 5

            f.write(str(class_label))
            f.write(' ')  
            for feat_term in featureids_terms:
                #print(feat_term)
                feat_term = feat_term.replace('(','').replace(')','')
                feat,term = feat_term.split(', ')[0],feat_term.split(', ')[1]
                term = term.replace("'", '')
                if term in v:
                    #tf = round((v.count(term))/float(len(v.split(' '))),4)    #normalized term frequency of a term in a document
                    tf_score = tf_compute(v,term)
                    try:
                        nDocs_term = len(inv_index[term])
                        #nDocs = os.system("ls subdir | wc -l")
                        nDocs = nDocs_in_subdir(k)   #number of docs(files) in a subdirectory a document belongs to
                        idf_score = idf_compute(nDocs, nDocs_term)   #100 is number of documents in a single newsgroup
                        #idf_score = idf(100, nDocs_term)   #100 is number of documents in a single newsgroup
                    except:
                        idf_score = 1.0

                    tf_idf_score = round(tf_score * idf_score, 2)
                    feat_tf_idf_map = str(feat)+':'+str(tf_idf_score)

                    #feat_tf_map = str(feat)+':'+str(tf)

                    f.write(feat_tf_idf_map)
                    f.write(' ')
                    doc_list[class_label].append(feat_tf_idf_map)
                    #doc_list[k].append(feat_tf_map)
            f.write('\n')


#main method
if __name__ == '__main__':
    
    #instantiate class 'Index'
    index = Index(nltk.word_tokenize, 
              EnglishStemmer(), 
              nltk.corpus.stopwords.words('english'))
    inv_index = index.indexed_docs()   #saves the inverted index into a variable

    #reads arguments from command line
    dir_newsgroups_data = sys.argv[1]    #reads directory of newsgroups data(which is the root directory mini_newsgroup)
    subject_body_index(dir_newsgroups_data)

    feature_defn_file = sys.argv[2]    #argument name to use to write feature definition file
    feature_defn_gen(feature_defn_file)
    print('Produced feature definition file')

    class_defn_file = sys.argv[3]    #argument name to use to write class definition file
    class_defn_gen(class_defn_file,dir_newsgroups_data)
    print('Produced class definition file')


    train_data_file = sys.argv[4]

    featureids_terms = feature_id_gen(feature_defn_file)
    class_definition_file = class_defn_pair_gen(class_defn_file)
    
    #generates either term-frequency based, inverse document frequency based or TF-IDF based training dataset
    #file 
    if '.TF' in train_data_file:
        training_data_tf_gen(train_data_file,featureids_terms,class_definition_file)
    elif '.IDF' in train_data_file:
        training_data_idf_gen(train_data_file,inv_index,featureids_terms,class_definition_file)
    elif '.TFIDF' in train_data_file:
        training_data_tf_idf_gen(train_data_file,inv_index,featureids_terms,class_definition_file)
    else:
        print("Use file names with extensions '.TF', '.IDF' or '.TFIDF' ")

    print('Produced training data file')
    

  



