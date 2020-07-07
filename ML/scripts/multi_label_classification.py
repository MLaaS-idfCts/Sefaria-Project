# GAMEPLAN
'''

'''

# IMPORT LIBRARIES
import sys
import time
import scipy
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm
from classes import DataManager
from datetime import datetime
from sklearn.svm import SVC, LinearSVC
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.adapt import BRkNNaClassifier, MLkNN
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from sklearn.feature_extraction.text import TfidfVectorizer


# choose convenient settings
# ignore warnings regarding column assignment, 
# e.g. df['col1'] = list1 -- not 100% sure about this
pd.options.mode.chained_assignment = None  # default='warn'

# do not limit num of rows in df to display
pd.set_option('display.max_rows', None)

# disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# width of column to dispaly in dataframe
pd.options.display.max_colwidth = 40

# location of actual csv
DATA_PATH = '/persistent/Sefaria-Project/ML/data/yishai_data.csv'

# dict of classifier types
classifiers = [
    # 1
    # OneVsRestClassifier(LinearSVC()),
    # OneVsRestClassifier(MultinomialNB()),
    # OneVsRestClassifier(GaussianNB()),
    # OneVsRestClassifier(LogisticRegression()),
    # OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.1, penalty='l2'), n_jobs=-1),
    OneVsRestClassifier(SGDClassifier()),
    # 2
    # BinaryRelevance(classifier=LinearSVC(),require_dense=[True, True]),
    # BinaryRelevance(classifier=MultinomialNB(),require_dense=[True, True]),
    # BinaryRelevance(classifier=SVC(),require_dense=[True, True]),
    # 3
    # LabelPowerset(classifier=LinearSVC(),require_dense=[True, True]),
    # 4
    # ClassifierChain(classifier=LinearSVC(),require_dense=[True, True]),
    # 5
    # (classifier=LinearSVC(),require_dense=[True, True]),
    ]

# ***VARIABLE INPUTS***
# how many topics to consider
NUM_TOPICS = 5
# max num of passages to examine
ROW_LIMIT = 100000
print(f"{NUM_TOPICS} topics and {ROW_LIMIT} for row limit.")

# load original data 
df = pd.read_csv(DATA_PATH)[:ROW_LIMIT]

# init class to manage data
data = DataManager(raw_df = df, num_topics = NUM_TOPICS, should_clean = True, should_stem = True)

# list of most commonly occurring topics
top_topics = data._get_top_topics()

# how many passages belong to each topic
topic_counts = data.get_topic_counts()
print(topic_counts.head())

# data in usable fromat, e.g. cleaned, stemmed, etc.
# e.g. should have column for passage text, and for each topic
data = data.preprocess_dataframe()

# randomly split into training and testing sets
train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)

# init column for topic predictions
test['pred_topics'] = None

# select just the words of each passage
train_text = train['passage_text']
test_text = test['passage_text']

# init a vectorizer that will convert string of words into numerical format
# vectorizer = TfidfVectorizer(max_features=5000,stop_words='english')
vectorizer = TfidfVectorizer(ngram_range=(1,2), 
# tokenizer=tokenize,
min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,smooth_idf=1, sublinear_tf=1 )
# vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')

# document-term matrix, i.e. numerical version of passage text
# Note: We only fit with training data, 
x_train = vectorizer.fit_transform(train_text)
# but NOT with testing data, because testing data should be "UNSEEN"
x_test = vectorizer.transform(test_text)

# topics columns, with 0/1 indicating if the topic of this column relates to that row's passage
y_train = train.drop(labels = ['passage_text','true_topics'], axis=1)
y_test = test.drop(labels = ['passage_text','true_topics'], axis=1)

# loop thru various classifier types
for classifier in classifiers:

    # for timing processes
    start_time = datetime.now()
    
    # train the classifier
    try:
        classifier.fit(x_train, y_train)
    except:
        y_train = y_train.values.toarray()
        # y_train = y_train.values.todense()
        # try:
        classifier.fit(
            x_train, 
            y_train
            )
    # except:


    # make predictions
    test_predictions = classifier.predict(x_test)
    train_predictions = classifier.predict(x_train)

    # convert csc matrix into list of csr matrices, e.g. preds_list = [[0,1,0,1,0],[0,0,0,1,0],...,[1,0,0,0,0]]
    # one matrix per passage, e.g. [0,0,0,1,1] inidicates this passage corrseponds to the last two topics
    preds_list = list(test_predictions)
    
    # init list of sublists, one sublist per passage, 
    # e.g. pred_labels_list = [['prayer'],['prayer','judges'],['moses']]
    pred_labels_list = []
    
    # loop thru each matrix in list, again one matrix per passage, 
    for array in preds_list:
    
        if isinstance(array, scipy.sparse.csr.csr_matrix) or isinstance(array, np.int64) or isinstance(array, scipy.sparse.lil.lil_matrix):
            # array = array.tolil().data.tolist()
            array = [array[0,i] for i in range(array.shape[1])]
    
        # init topics list for this row, e.g. passage_labels = ['prayer', 'moses']
        passage_labels = []
    
    
        # if 1 occurs in ith element in the array, record ith topic
        for topic_index, pred_value in enumerate(list(array)):
            if pred_value != 0:
                passage_labels.append(top_topics[topic_index])
        pred_labels_list.append(passage_labels)

    test['pred_topics'] = pred_labels_list

    cols=['passage_text','true_topics','pred_topics']

    result = test[cols]
    topics_comparison = test[['true_topics','pred_topics']]
    # print(topics_comparison)
    print('\n*******************************************************************')
    print('Classifier type:',classifier)

    true_label_lists = result.true_topics.tolist()
    pred_label_lists = result.pred_topics.tolist()

    assert len(true_label_lists) == len(pred_label_lists)

    num_passages = len(true_label_lists)

    y_true = []
    y_pred = []

    for i in range(num_passages):

        true_label_list = []
        pred_label_list = []
        
        try:
            true_label_list = true_label_lists[i]
        except:
            pass
        
        try:
            pred_label_list = pred_label_lists[i]
        except:
            pass

        # 0) NULL CASE --> No true or pred labels 
        if len(pred_label_list) == 0 and len(pred_label_list) == 0:
                y_true.append('None')
                y_pred.append('None')
    
        # 1) MATCH --> true label == pred label 
        for true_label in true_label_list:
            if true_label in pred_label_list:
                y_true.append(true_label)
                y_pred.append(true_label)

        # 2) FALSE NEGATIVE --> true label was not predicted
            else:
                y_true.append(true_label)
                y_pred.append("None")

        # 3) FALSE POSITIVE --> pred label was not true
        for pred_label in pred_label_list:
            if pred_label not in true_label_list:
                y_true.append("None")
                y_pred.append(pred_label)
    
    topics_list = ['None'] + top_topics
            
    y_actu = pd.Categorical(y_true, categories=topics_list)
    y_pred = pd.Categorical(y_pred, categories=topics_list)

    cm = pd.crosstab(y_actu, y_pred, rownames=['True'], colnames = ['Pred'], dropna=False) 
    # print(cm)
    cm = cm.values

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    
    np.seterr(divide='ignore', invalid='ignore')

    # precision = how many predicted were correct
    precision = [round(num,2) for num in TP/(TP+FP)]
    
    # recall = how many correct were predicted
    recall = [round(num,2) for num in TP/(TP+FN)]

    score_df = pd.DataFrame(
        {'Topics': topics_list,'Precision': precision, 'Recall': recall}
        ).sort_values(by=['Precision', 'Recall'], ascending=False)

    print("Time taken:", datetime.now() - start_time)
    
    print(score_df.head())

print()