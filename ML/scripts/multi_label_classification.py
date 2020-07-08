# IMPORTS
import sys
import time
import scipy
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm
from classes import DataManager, ConfusionMatrix, Predictor, DataConverter, Scorer
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


# for timing processes
start_time = datetime.now()
print("Time taken for :\n", datetime.now() - start_time)
start_time = datetime.now()

# SETTINGS
# ignore warnings regarding column assignment, e.g. df['col1'] = list1 -- not 100% sure about this
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
    BinaryRelevance(classifier=LinearSVC(),require_dense=[True, True]),
    # LabelPowerset(classifier=LinearSVC(),require_dense=[True, True]),
    # ClassifierChain(classifier=LinearSVC(),require_dense=[True, True]),
    ]

# count passages with no topics
COUNT_NONE = True
# how many topics to consider
NUM_TOPICS = 5
# max num of passages to examine
ROW_LIMIT = 50000
print(f"{NUM_TOPICS} topics and {ROW_LIMIT} for row limit.")


# LOAD AND PROCESS DATA

# load original data 
df = pd.read_csv(DATA_PATH)[:ROW_LIMIT]

# init class to manage data
data = DataManager(raw_df = df, num_topics = NUM_TOPICS, should_clean = True, should_stem = True, count_none = COUNT_NONE)

# list of most commonly occurring topics
top_topics = data._get_top_topics()

# how many passages belong to each topic
topic_counts = data.get_topic_counts()

# data in usable fromat, e.g. cleaned, stemmed, etc.
# e.g. should have column for passage text, and for each topic
data = data.preprocess_dataframe()

print("Time taken for preprocessing data:\n", datetime.now() - start_time)
start_time = datetime.now()

print("Time taken for preprocessing data:\n", datetime.now() - start_time)
start_time = datetime.now()

x_train, x_test, y_train, y_test, test = DataConverter(data).get_datasets()
print("Time taken for data converter:\n", datetime.now() - start_time)
start_time = datetime.now()

# loop thru various classifier types
for classifier in classifiers:
    print('\n*******************************************************************\nClassifier:',classifier)
    
    # TRAIN
    try:
        classifier.fit(x_train, y_train)
    except:
        y_train = y_train.values.toarray()
        classifier.fit(x_train, y_train)
    print("Time taken for training:\n", datetime.now() - start_time)
    start_time = datetime.now()

    # PREDICT
    pred_vs_true = Predictor(classifier, test, x_test, top_topics).get_result()
    print("Time taken for prediction:\n", datetime.now() - start_time)
    start_time = datetime.now()

    # CONFUSION MATRIX
    cm = ConfusionMatrix(pred_vs_true, top_topics).get_values()
    print("Time taken for confusion matrix:\n", datetime.now() - start_time)
    start_time = datetime.now()

    # SCORE MODEL
    score_df = Scorer(cm, top_topics, topic_counts).get_result()
    print("Time taken for scorer:\n", datetime.now() - start_time)
    start_time = datetime.now()


    print(score_df.round(2))


print()


# RANK MODELS