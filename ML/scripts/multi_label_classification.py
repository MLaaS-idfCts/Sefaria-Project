import sys
import time
import scipy
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm
from classes import DataManager, ConfusionMatrix, Predictor, DataConverter, Scorer, Trainer
from datetime import datetime
from sklearn.svm import SVC, LinearSVC
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# from personal_settings import *
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.adapt import BRkNNaClassifier, MLkNN
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from sklearn.feature_extraction.text import TfidfVectorizer

def time_and_reset(start_time):
    """
    Usage: To print the time elapsed since previous call, type:
    start_time = time_and_reset(start_time)
    """
    print(datetime.now() - start_time)
    return datetime.now()


start_time = datetime.now()

start_time = time_and_reset(start_time)

parameters = [
    {'classifier': [MultinomialNB()],'classifier__alpha': [0.8, 1.1],},
    {'classifier': [SVC()],'classifier__kernel': ['rbf', 'linear'],},
    ]

DATA_PATH = '/persistent/Sefaria-Project/ML/data/yishai_data.csv'

classifiers = [
    BinaryRelevance(classifier=LinearSVC()),
    ]

# count passages with no topics
COUNT_NONE = False

# how many topics to consider
NUM_TOPICS = 5

# max num of passages to examine
ROW_LIMIT = 1000
print(f"{NUM_TOPICS} topics and {ROW_LIMIT} for row limit.")

# raw dataframe
df = pd.read_csv(DATA_PATH)[:ROW_LIMIT]

# preprocessed data
data = DataManager(raw_df = df, num_topics = NUM_TOPICS, should_clean = True, should_stem = True, count_none = COUNT_NONE)

# list of most commonly occurring topics
top_topics = data._get_top_topics()

# how many passages belong to each topic
topic_counts = data.get_topic_counts()
print(topic_counts)

# data in usable fromat, e.g. cleaned, stemmed, etc.
# e.g. should have column for passage text, and for each topic
data = data.preprocess_dataframe()

# init a vectorizer that will convert string of words into numerical format
vectorizer = TfidfVectorizer(
    # norm = 'l2',
    # min_df = 3, 
    # max_df = 0.9, 
    # use_idf = 1,
    # analyzer = 'word', 
    # stop_words = 'english',
    # smooth_idf = 1, 
    # ngram_range = (1,3),
    # max_features = 10000,
    # sublinear_tf = 1,
    # strip_accents = 'unicode', 
    )

x_train, x_test, y_train, y_test, test = DataConverter(data).get_datasets(vectorizer)

predictors = []

# loop thru various classifier types
for classifier in classifiers:

    classifier = Trainer(classifier).train(x_train, y_train)
    
    predictor = Predictor(classifier, test, top_topics)
    
    predictors.append(predictor)

if __name__ == "__main__":

    for predictor in predictors:

        preds_list = predictor.get_preds_list(x_test)

        pred_vs_true = predictor.get_pred_vs_true(preds_list)

        cm = ConfusionMatrix(pred_vs_true, top_topics).get_values()

        score_df = Scorer(cm, top_topics, topic_counts).get_result()

        print('Overall F1score:',score_df.loc['OVERALL','F1score'].round(5))

    end_time = datetime.now()

    total_time = end_time - start_time
    
    print(total_time)

    print()


    # RANK MODELS

