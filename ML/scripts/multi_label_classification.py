import sys
import time
import scipy
import numpy as np
import pickle
import pandas as pd
import warnings

from tqdm import tqdm
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
    print(datetime.now() - start_time)
    return datetime.now()
    
# ignore warnings regarding column assignment, e.g. df['col1'] = list1 -- not 100% sure about this
pd.options.mode.chained_assignment = None  # default='warn'

# do not limit num of rows in df to display
pd.set_option('display.max_rows', None)

# disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# width of column to dispaly in dataframe
pd.options.display.max_colwidth = 40


start_time = datetime.now()

# start_time = time_and_reset(start_time)

parameters = [
    {'classifier': [MultinomialNB()],'classifier__alpha': [0.8, 1.1],},
    {'classifier': [SVC()],'classifier__kernel': ['rbf', 'linear'],},
    ]

# DATA_PATHS = []

# DATA_PATH = '/persistent/Sefaria-Project/ML/data/multiversion.csv'
# DATA_PATHS.append(DATA_PATH)
# DATA_PATH = '/persistent/Sefaria-Project/ML/data/yishai_data.csv'
# DATA_PATHS.append(DATA_PATH)
DATA_PATH = '/persistent/Sefaria-Project/ML/data/multi_version_english.csv'
# DATA_PATHS.append(DATA_PATH)

# for DATA_PATH in DATA_PATHS:

print(DATA_PATH[DATA_PATH.rfind('data'):])

classifiers = [
    BinaryRelevance(classifier=LinearSVC()),
    # GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy'),
    ]

# count passages with no topics
COUNT_NONE = True

# how many topics to consider
NUM_TOPICS = 20

# max num of passages to examine
ROW_LIMITS = {
    # 0:1000,
    1:10000,
    # 2:20000,
    # 3:40000,
    # 4:80000,
    # 5:120000,
    # 6:180000,
}

# max num of nones to allow, e.g. 1/8 means to allow 
# only enough nones so that they will number no more 
# than 1/8 of the passages with non-trivial topic
none_ratios = {
    0:0.2,
    1:0.4,
    2:0.6,
    3:0.8,
    4:1.0,
    5:1.2,
    6:1.4,
    7:1.6,
    8:1.8,
    9:2.0,
    # 7:5,
    # 8:10,
    # 9:20,
}

row_lim = 40000

for expt_num, none_ratio in none_ratios.items():
# for expt_num, row_lim in ROW_LIMITS.items():

    
    # raw dataframe
    raw_df = pd.read_csv(DATA_PATH).sample(row_lim)
    # raw_df = pd.read_csv(DATA_PATH)[:ROW_LIMIT]

    # check shape
    # print("Raw shape:",raw_df.shape)

    # preprocessed data
    data = DataManager(
        raw_df = raw_df, 
        none_ratio = none_ratio, 
        num_topics = NUM_TOPICS, 
        should_stem = True, 
        # should_clean = False, 
        should_remove_stopwords = True, 
        # count_none = COUNT_NONE,
        # none_ratio = 10,
        )


    # list of most commonly occurring topics
    top_topics = data._get_top_topics()

    # how many passages belong to each topic
    topic_counts = data.get_topic_counts()
    # print(topic_counts)

    # data in usable fromat, e.g. cleaned, stemmed, etc.
    # e.g. should have column for passage text, and for each topic
    data = data.preprocess_dataframe()

    # check shape
    # print("Processed shape:",data.shape)

    data.to_csv(DATA_PATH[:-4]+'processed.csv')


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

    splitter = DataSplitter(data)
    train, test, x_train, x_test, y_train, y_test = splitter.get_datasets(vectorizer)

    predictors = []

    # loop thru various classifier types
    for classifier in classifiers:

        classifier = Trainer(classifier).train(x_train, y_train)
        
        predictor = Predictor(classifier, top_topics)
        # predictor = Predictor(classifier, train, top_topics)
        
        predictors.append(predictor)

    # if __name__ == "__main__":

        for predictor in predictors:

            test_pred_list = predictor.get_preds_list(x_test)
            train_pred_list = predictor.get_preds_list(x_train)

            train_pred_vs_true = predictor.get_pred_vs_true(train, train_pred_list)
            test_pred_vs_true = predictor.get_pred_vs_true(test, test_pred_list)

            train_pred_vs_true.to_pickle(f'/persistent/Sefaria-Project/ML/data/train_pred_vs_true_{none_ratio}.pkl')
            test_pred_vs_true.to_pickle(f'/persistent/Sefaria-Project/ML/data/test_pred_vs_true_{none_ratio}.pkl')

            train_cm = ConfusionMatrix(train_pred_vs_true, top_topics, 
                # should_print = True
                )

            test_cm = ConfusionMatrix(test_pred_vs_true, top_topics, 
                # should_print = True
                )
            
            train_cm = train_cm.get_values()
            test_cm = test_cm.get_values()

            train_cm.dump(f"/persistent/Sefaria-Project/ML/data/train_cm_{none_ratio}.dat")
            test_cm.dump(f"/persistent/Sefaria-Project/ML/data/test_cm_{none_ratio}.dat")
            # cm.dump(f"/persistent/Sefaria-Project/ML/data/cm_{row_lim}.dat")
            # cm = numpy.load("cm_{row_lim}.dat")

            # print(cm)

            scorer = Scorer(top_topics, topic_counts, row_lim, expt_num, none_ratio)
 
            train_score_df = scorer.get_result(train_cm, dataset = 'train')
            test_score_df = scorer.get_result(test_cm, dataset = 'test')
 
            train_score_df.to_pickle(f'/persistent/Sefaria-Project/ML/data/train_score_df{none_ratio}.pkl')
            test_score_df.to_pickle(f'/persistent/Sefaria-Project/ML/data/test_score_df_{none_ratio}.pkl')

            # print(score_df.round(2))
            # print('Overall F1score:',score_df.loc['OVERALL','F1score'].round(5))

        
        end_time = datetime.now()

        total_time = end_time - start_time
        
        print("# Time taken:", total_time)

    # RANK MODELS

