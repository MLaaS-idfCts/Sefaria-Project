import sys
import time
import scipy
import numpy as np
import pickle
import pandas as pd
import warnings

from tqdm import tqdm
from numpy import arange
from classes import DataManager, ConfusionMatrix, Predictor, DataSplitter, Scorer, Trainer
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
    print('#',datetime.now() - start_time)
    return datetime.now()
    
# ignore warnings regarding column assignment, e.g. df['col1'] = list1 -- not 100% sure about this
pd.options.mode.chained_assignment = None  # default='warn'

# do not limit num of rows in df to display
pd.set_option('display.max_rows', None)

# disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# width of column to dispaly in dataframe
pd.options.display.max_colwidth = 35


start_time = datetime.now()

# start_time = time_and_reset(start_time)

parameters = [
    {'classifier': [MultinomialNB()],'classifier__alpha': [0.8, 1.1],},
    {'classifier': [SVC()],'classifier__kernel': ['rbf', 'linear'],},
    ]

DATA_PATH = '/persistent/Sefaria-Project/ML/data/multi_version_english.csv'

classifiers = [
    BinaryRelevance(classifier=LinearSVC()),
    # GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy'),
    ]

# Ratio of nones compared with num of occurrences of top topic.
# For example the first is 0.3 which means the max nones we allow 
# is 30% of the total occurrences of the top topic.

none_ratioes = list(arange(0.2, 2.2, 0.2))

row_lims = {
    0:250,
    1:500,
    2:1000,
    3:2500,
    4:5000,
    5:10000,
    6:25000,
    7:50000,
    8:100000,
    9:180000,
}

row_lim = 5000
print("row_lim =",row_lim)
print(f"# expt_nums:none_ratio {[f'{i}:{round(none_ratio,1)}' for i,none_ratio in enumerate(none_ratioes)]}")

# how many topics to consider
NUM_TOPICS = 3

# for expt_num, row_lim in tqdm(row_lims.items()):
# for expt_num, none_ratio in tqdm(none_ratios.items()):
# for expt_num, none_ratio in enumerate(none_ratioes):
if True:
    expt_num = 0
    none_ratio = 0.5
    
    start_time = time_and_reset(start_time)
    
    # raw dataframe
    raw_df = pd.read_csv(DATA_PATH).sample(row_lim)

    # preprocessed data
    data = DataManager(
        
        raw_df = raw_df, 
        num_topics = NUM_TOPICS, 

        # how many nones to keep 
        none_ratio = none_ratio, # same as top topic
        # none_ratio = none_ratio, 
        # none_ratio = 'all', 

        # how to modify passage text
        should_stem = False, 
        should_clean = True, 
        should_remove_stopwords = False, 
        )


    # list of most commonly occurring topics
    reduced_topics_df = data.get_reduced_topics_df()

    limited_nones_df = data.limit_nones(reduced_topics_df)

    one_hot_encoded_df = data.one_hot_encode(limited_nones_df)

    tidied_up_df = data.tidy_up(one_hot_encoded_df)

    tidied_up_df.to_csv(DATA_PATH[:-4] + 'tidied_up_df.csv')

    data_df = tidied_up_df

    # init a vectorizer that will convert string of words into numerical format
    vectorizer = TfidfVectorizer()

    # init class to split data
    splitter = DataSplitter(data_df)
    # get subdivided datasets
    train, test, x_train, x_test, y_train, y_test = splitter.get_datasets(vectorizer)

    # init series of predictors, one for each classifier
    predictors = []

    # loop thru various classifier types
    for classifier in classifiers:

        trainer = Trainer(classifier)
        classifier = trainer.train(x_train, y_train)
        
        # init class of predictor based on classifier and list of chosen topics
        predictor = Predictor(classifier, data.ranked_topic_names_without_none)
        
        # store predictor in arsenal
        predictors.append(predictor)

        # loop through all predictors
        for predictor in predictors:

            # list of predictions
            train_pred_list = predictor.get_preds_list(x_train)
            test_pred_list = predictor.get_preds_list(x_test)

            # columns to compare pred and true labels
            train_pred_vs_true = predictor.get_pred_vs_true(train, train_pred_list)
            test_pred_vs_true = predictor.get_pred_vs_true(test, test_pred_list)

            # save results
            train_pred_vs_true.to_pickle(f'/persistent/Sefaria-Project/ML/data/train_pred_vs_true_{none_ratio}.pkl')
            test_pred_vs_true.to_pickle(f'/persistent/Sefaria-Project/ML/data/test_pred_vs_true_{none_ratio}.pkl')

            # init class to constrcut confusion matrix
            cm_maker = ConfusionMatrix(
                                        data.ranked_topic_names_with_none, 
                                        # should_print = True
                                        )

            # get confusion matrices
            train_cm = cm_maker.get_cm_values(train_pred_vs_true)
            test_cm = cm_maker.get_cm_values(test_pred_vs_true)

            # init class to compute scores
            scorer = Scorer(data.ranked_topic_names_with_none, data.ranked_topic_counts_with_none, row_lim, expt_num, none_ratio, 
                            # should_print = True
                            )
 
            # get actual scores 
            train_score_df = scorer.get_stats_df(train_cm, dataset = 'train')
            test_score_df = scorer.get_stats_df(test_cm, dataset = 'test')
 
            # save results
            train_score_df.to_pickle(f'/persistent/Sefaria-Project/ML/data/train_score_df{none_ratio}.pkl')
            test_score_df.to_pickle(f'/persistent/Sefaria-Project/ML/data/test_score_df_{none_ratio}.pkl')

# compute and display time elapsed
end_time = datetime.now()
total_time = end_time - start_time
print("# Total time taken:", total_time)
print()

    # RANK MODELS

