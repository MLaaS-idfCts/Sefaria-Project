import mpu
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
pd.options.display.max_colwidth = 50


start_time = datetime.now()

# start_time = time_and_reset(start_time)

parameters = [
    {'classifier': [MultinomialNB()],'classifier__alpha': [0.8, 1.1],},
    {'classifier': [SVC()],'classifier__kernel': ['rbf', 'linear'],},
    ]

DATA_PATHS = []

DATA_PATH = 'data/concat_english_prefix_hebrew.csv'
DATA_PATHS.append(DATA_PATH)

# DATA_PATH = 'data/multi_version_english.csv'
# DATA_PATHS.append(DATA_PATH)



classifiers = [
    BinaryRelevance(classifier=LinearSVC()),
    ]

row_lim = 1000
# row_lim = N
print("row_lim =",row_lim)

# how many topics to consider
NUM_TOPICS = 2
print("# num_topics =", NUM_TOPICS)

# implement_rules = [True,False]
implement_rules = False

use_cached_df = False
# use_cached_df = True

# for expt_num, none_ratio in enumerate(none_ratioes):
# for expt_num, none_ratio in enumerate(implement_rules):

# which language(s) do you want to vectorize
# langs_to_vec = ['eng','heb','both']
lang_to_vec = 'eng'

should_separate = True
# separate_options = [True,False]

for expt_num, DATA_PATH in enumerate(DATA_PATHS):
# for expt_num, lang_to_vec in enumerate(langs_to_vec):
# for expt_num, should_separate in enumerate(separate_options):

# if True:
    # expt_num = 0

    print(f'\n\n# expt #{expt_num} = {DATA_PATH}')

    
    none_ratio = 1.1
    
    start_time = time_and_reset(start_time)
    
    if not use_cached_df:

        # shuffle
        # raw_df = pd.read_csv(DATA_PATH).sample(frac=1)

        # take subportion
        raw_df = pd.read_csv(DATA_PATH)[:row_lim]
        # raw_df = pd.read_csv(DATA_PATH)

        print("# actual num rows taken =",raw_df.shape[0])


        # preprocessed data
        data = DataManager(raw_df = raw_df, num_topics = NUM_TOPICS, none_ratio = none_ratio, 
                            should_stem = False, should_clean = True, should_remove_stopwords = False, )

        # list of most commonly occurring topics
        ontology_counts_dict = data.get_ontology_counts_dict()

        # pickle.dump(ontology_counts_dict, open(f'data/ontology_counts_dict_row_lim_{row_lim}_file.pkl', "wb"))  # save it into a file named save.p

        # mpu.io.write(f'data/ontology_counts_dict_row_lim_{row_lim}_file.pkl', ontology_counts_dict)

        with open('data/ontology_counts_dict.pkl', 'wb') as handle:
            pickle.dump(ontology_counts_dict, handle, 
                protocol=3
                # protocol=pickle.HIGHEST_PROTOCOL
            )


        # list of most commonly occurring topics
        reduced_topics_df = data.get_reduced_topics_df()

        limited_nones_df = data.limit_nones(reduced_topics_df)

        one_hot_encoded_df = data.one_hot_encode(limited_nones_df)

        tidied_up_df = data.tidy_up(one_hot_encoded_df)

        tidied_up_df.to_csv(DATA_PATH[:-4] + '_tidied_up_df.csv')
        

    elif use_cached_df:

        tidied_up_df = pd.read_csv(DATA_PATH[:-4] + '_tidied_up_df.csv')


    data_df = tidied_up_df

    # combine english and hebrew
    if lang_to_vec == 'eng':
        data_df['passage_words'] = data_df['passage_text_english']

    if lang_to_vec == 'heb':
        data_df['passage_words'] = data_df['passage_text_hebrew_parsed']

    if lang_to_vec == 'both':
        data_df['passage_words'] = data_df['passage_text_english'] + ' ' + data_df['passage_text_hebrew_parsed'] 

    # init a vectorizer that will convert string of words into numerical format
    vectorizer = TfidfVectorizer()

    # init class to split data
    splitter = DataSplitter(data_df, should_separate, DATA_PATH = DATA_PATH)

    # get subdivided datasets
    train, test, x_train, x_test, y_train, y_test = splitter.get_datasets(vectorizer)

    # init series of predictors, one for each classifier
    predictors = []

    # loop thru various classifier types
    for classifier in classifiers:

        trainer = Trainer(classifier)
        classifier = trainer.train(x_train, y_train)
        
        # init class of predictor based on classifier and list of chosen topics
        predictor = Predictor(classifier,  implement_rules = implement_rules, top_topic_names = data.ranked_topic_names_without_none)
        
        # store predictor in arsenal
        predictors.append(predictor)

        # loop through all predictors
        for predictor in predictors:

            # list of predictions
            train_pred_list = predictor.get_preds_list(x_train, text_df = train)
            test_pred_list = predictor.get_preds_list(x_test, text_df = test)

            # columns to compare pred and true labels
            train_pred_vs_true = predictor.get_pred_vs_true(train, train_pred_list)
            test_pred_vs_true = predictor.get_pred_vs_true(test, test_pred_list)

            # save results
            train_pred_vs_true.to_pickle(f'/persistent/Sefaria-Project/ML/data/train_pred_vs_true_{none_ratio}.pkl')
            test_pred_vs_true.to_pickle(f'/persistent/Sefaria-Project/ML/data/test_pred_vs_true_{none_ratio}.pkl')

            # init class to constrcut confusion matrix
            cm_maker = ConfusionMatrix(data.ranked_topic_names_with_none, 
                                        # should_print = True
                                        )

            # get confusion matrices
            test_cm = cm_maker.get_cm_values(test_pred_vs_true)
            train_cm = cm_maker.get_cm_values(train_pred_vs_true)

            # check the worst performing examples to see what's going wrong
            worst_test = cm_maker.check_worst(test_cm, test_pred_vs_true)
            worst_train = cm_maker.check_worst(train_cm, train_pred_vs_true)

            # init class to compute scores
            scorer = Scorer(data.ranked_topic_names_without_none, data.ranked_topic_counts_without_none, row_lim, expt_num, none_ratio, 
            # scorer = Scorer(data.ranked_topic_names_with_none, data.ranked_topic_counts_with_none, row_lim, expt_num, none_ratio, 
                            should_print = True
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
print("\n# Total time taken:", total_time)
print()

    # RANK MODELS

