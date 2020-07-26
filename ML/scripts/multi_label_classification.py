# imports 
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
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.adapt import BRkNNaClassifier, MLkNN
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from sklearn.feature_extraction.text import TfidfVectorizer


# functions
def time_and_reset(start_time):
    print('#',datetime.now() - start_time)
    return datetime.now()

    
# settings

# ignore warnings regarding column assignment, e.g. df['col1'] = list1 -- not 100% sure about this
pd.options.mode.chained_assignment = None  # default='warn'

# do not limit num of rows in df to display
pd.set_option('display.max_rows', None)

# disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# width of column to dispaly in dataframe
pd.options.display.max_colwidth = 50


# actual code begins

start_time = datetime.now()

DATA_PATH = 'data/concat_english_prefix_hebrew.csv'

classifiers = [BinaryRelevance(classifier=LinearSVC()),]

row_lims = [20000,40000,80000,100000,170000, None]

# list of topics that you want to train and analyze
chosen_topics_list = []
chosen_topics = ['generically-dependent-continuant', 'independent-continuant', 'occurent', 'specifically-dependent-continuant']
chosen_topics_list.append(chosen_topics)
chosen_topics = ['generically-dependent-continuant', 'independent-continuant', 'occurent', 'quality', 'realizable-entity']
chosen_topics_list.append(chosen_topics)

# how many topics to consider
topic_limit = None
print("# topic_limit =", topic_limit)

# whether to use rule based logic after machine learning algorithm
implement_rules = False

# use cleaned stored dataframe, instead of always reprocessing it
use_cached_df = False

# which language(s) do you want to vectorize
# langs_to_vec = ['eng','heb','both']
lang_to_vec = 'eng'

# compute number of children for each node
get_ontology_counts = False

# do you want to train on all nodes of ontology
use_expanded_topics = True
# use_expanded_topics = False

# keep training examples from leaking into the test set
should_separate = True

for expt_num, chosen_topics in enumerate(chosen_topics_list):
# for expt_num, row_lim in enumerate(row_lims):
# if True:
    row_lim = 40000

    print(f'\n\n# expt #{expt_num} = {row_lim}')

    # use number of nones that slightly greater than the most prevalent topic
    none_ratio = 1.1
    
    if not use_cached_df:

        # shuffle
        raw_df = pd.read_csv(DATA_PATH).sample(frac=1)

        # take subportion
        raw_df = raw_df[:row_lim]
        print("# actual num rows taken =",raw_df.shape[0])

        # preprocessed data
        data = DataManager(raw_df = raw_df, topic_limit = topic_limit, none_ratio = none_ratio, 
                            should_stem = False, should_clean = True, should_remove_stopwords = False, 
                            use_expanded_topics = use_expanded_topics, chosen_topics = chosen_topics
                            )


        if get_ontology_counts:

            # capture list of most commonly occurring topics
            ontology_counts_dict = data.get_ontology_counts_dict()

            # store result
            with open(f'data/ontology_counts_dict_row_lim_{row_lim}.pickle', 'wb') as handle:
                pickle.dump(ontology_counts_dict, handle, protocol=3)


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
        predictor = Predictor(classifier, implement_rules = implement_rules, top_topic_names = data.ranked_topic_names_without_none)
        
        # store predictor in arsenal
        predictors.append(predictor)

        # init class to constrcut confusion matrix
        cm_maker = ConfusionMatrix(data.ranked_topic_names_with_none, 
                                    # should_print = True
                                    )

        # init class to compute scores
        scorer = Scorer(data.ranked_topic_names_without_none, data.ranked_topic_counts_without_none, row_lim, expt_num, none_ratio, 
        # scorer = Scorer(data.ranked_topic_names_with_none, data.ranked_topic_counts_with_none, row_lim, expt_num, none_ratio, 
                        should_print = True,
                        use_expanded_topics = use_expanded_topics, chosen_topics = chosen_topics
                        )

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

            # get confusion matrices
            train_cm = cm_maker.get_cm_values(train_pred_vs_true)
            test_cm = cm_maker.get_cm_values(test_pred_vs_true)

            # check the worst performing examples to see what's going wrong
            worst_train = cm_maker.check_worst(train_cm, train_pred_vs_true)
            worst_test = cm_maker.check_worst(test_cm, test_pred_vs_true)

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

