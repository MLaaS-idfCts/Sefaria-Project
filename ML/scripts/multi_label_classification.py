from datetime import datetime
start_time = datetime.now()

# imports 
import sys
import time
import scipy
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import warnings

from tqdm import tqdm
from numpy import arange
from classes import DataManager, ConfusionMatrix, Predictor, DataSplitter, Scorer, Trainer, Categorizer, MultiStageClassifier
from matplotlib import pyplot as plt
from sklearn.svm import SVC, LinearSVC
from collections import Counter
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
# from multi_stage_classifier import multi_stage_classifier
from sklearn.model_selection import train_test_split, GridSearchCV
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from sklearn.feature_extraction.text import TfidfVectorizer


# **********************************************************************************************************
# functions
# **********************************************************************************************************

def time_and_reset(start_time):
    """
    Usage: 
    start_time = time_and_reset(start_time)
    """
    print('#',datetime.now() - start_time)
    return datetime.now()


# **********************************************************************************************************
# settings
# **********************************************************************************************************

# ignore warnings regarding column assignment, e.g. df['col1'] = list1 -- not 100% sure about this
pd.options.mode.chained_assignment = None  # default='warn'

# do not limit num of rows in df to display
pd.set_option('display.max_rows', None)

# disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# width of column to dispaly in dataframe
pd.options.display.max_colwidth = 50


# **********************************************************************************************************
# actual code begins
# **********************************************************************************************************

DATA_PATH = 'data/concat_english_prefix_hebrew.csv'

classifier = BinaryRelevance(classifier=LinearSVC())

vectorizer = TfidfVectorizer()

# list of topics that you want to train and analyze
super_topics_list = [
    ['occurent', 'specifically-dependent-continuant','independent-continuant','generically-dependent-continuant'],
    ['generically-dependent-continuant', 'independent-continuant', 'occurent', 'quality', 'realizable-entity']
]

# which language(s) do you want to vectorize
# langs_to_vec = ['eng','heb','both']
lang_to_vec = 'eng'

row_lim = 1000

for expt_num, super_topics in enumerate(super_topics_list):

    print('# expt_num',expt_num)
    print('# super_topics',super_topics)

    data = DataManager(data_path = DATA_PATH, row_lim = row_lim, 
                        # topic_limit = topic_limit, 
                        super_topics = super_topics,
                        lang_to_vec = lang_to_vec,
                        should_stem = False, should_clean = True, 
                        should_remove_stopwords = False)

    data.prepare_dataframe()    
    
    categorizer = Categorizer(data.df, super_topics)

    categorized_df = categorizer.sort_children(max_children = 3)

    topic_lists = categorizer.topic_lists


    predictor = Predictor(classifier = classifier, 
                            vectorizer = vectorizer, 
                            df = categorized_df,
                            super_topics = super_topics,
                            topic_lists = topic_lists,
                            )

    predictor.one_hot_encode()

    predictor.split_data()

    predictor.topic_group = 'Super Topics'

    predictor.fit_and_pred()


    for super_topic in super_topics:

        predictor.topic_group = f'Children of {super_topic}'

        predictor.fit_and_pred()


    train = predictor.train_set
    test = predictor.test_set


    super_topic_scores_dict = {}


    for super_topic in super_topics:

        cm_topics = categorizer.topic_lists[f'Children of {super_topic}'] + ['None']

        cm_maker = ConfusionMatrix(super_topic, cm_topics, expt_num)

        true_col = f'True Children of {super_topic}'
        pred_col = f'Pred Children of {super_topic}'

        train_pred_vs_true = train[[true_col,pred_col]]
        test_pred_vs_true = test[[true_col,pred_col]]

        train_cm = cm_maker.get_cm_values(train_pred_vs_true, data_set = 'train')
        test_cm = cm_maker.get_cm_values(test_pred_vs_true, data_set = 'test')

#     # check the worst performing examples to see what's going wrong
#     worst_train = cm_maker.check_worst(train_cm, train_pred_vs_true)
#     worst_test = cm_maker.check_worst(test_cm, test_pred_vs_true)

        scorer = Scorer(
            topic_counts=categorizer.topic_counts[super_topic],
            expt_num=expt_num, 
            super_topic=super_topic, 
            )

        # get actual scores 
        train_score_df = scorer.get_stats_df(train_cm, dataset = 'train')
        test_score_df = scorer.get_stats_df(test_cm, dataset = 'test')

        super_topic_scores = {
            'train':train_score_df.iloc[-1],
            'test':test_score_df.iloc[-1],
        }

        super_topic_scores_dict[super_topic] = super_topic_scores
    
    overall_fscore = {
        'train':0,
        'test':0
    }

    total_occurrences = 0

    for super_topic in super_topics:
    
        total_occurrences += super_topic_scores_dict[super_topic]['train']['Occurrences']


    for super_topic in super_topics:

        print('# super_topic',super_topic)

        proportion = super_topic_scores_dict[super_topic]['train']['Occurrences']/total_occurrences

        fscore_contribution_test = proportion*super_topic_scores_dict[super_topic]['test']['F1score']
        overall_fscore['test'] += fscore_contribution_test 

        fscore_contribution_train = proportion*super_topic_scores_dict[super_topic]['train']['F1score']
        overall_fscore['train'] += fscore_contribution_train 

        print('# test',round(fscore_contribution_test,2))
        print('# train',round(fscore_contribution_train,2))

    print('test', overall_fscore['test'])
    print('train', overall_fscore['train'])

print()


