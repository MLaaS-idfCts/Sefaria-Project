from datetime import datetime
start_time = datetime.now()

# imports 
import sys
import time
import scipy
import numpy as np
import pickle
import pandas as pd
import warnings

from tqdm import tqdm
from numpy import arange
from classes import DataManager, ConfusionMatrix, Predictor, DataSplitter, Scorer, Trainer, Categorizer
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
# from multi_stage_classifier import multi_stage_classifier
from sklearn.model_selection import train_test_split, GridSearchCV
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from sklearn.feature_extraction.text import TfidfVectorizer


# functions
def time_and_reset(start_time):
    """
    Usage: 
    start_time = time_and_reset(start_time)
    """
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


DATA_PATH = 'data/concat_english_prefix_hebrew.csv'

# classifiers = [BinaryRelevance(classifier=LinearSVC()),]
classifier = BinaryRelevance(classifier=LinearSVC())

# row_lims = [20000,40000,80000,100000,170000, None]

# list of topics that you want to train and analyze
chosen_super_topics_list = []
chosen_super_topics = ['generically-dependent-continuant', 'independent-continuant', 'occurent', 'specifically-dependent-continuant']
chosen_super_topics_list.append(chosen_super_topics)
# chosen_super_topics = ['generically-dependent-continuant', 'independent-continuant', 'occurent', 'quality', 'realizable-entity']
# chosen_super_topics_list.append(chosen_super_topics)

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

stages = [1,2]

# for expt_num, chosen_super_topics in enumerate(chosen_super_topics_list):
for expt_num, stage in enumerate(stages):
# for expt_num, row_lim in enumerate(row_lims):
# if True:
    row_lim = 100

    print(f'\n\n# expt #{expt_num} = {row_lim}')

    # use number of nones that slightly greater than the most prevalent topic
    none_ratio = 1.1
    
    if use_cached_df:

        tidied_up_df = pd.read_csv(DATA_PATH[:-4] + '_tidied_up_df.csv')

    elif not use_cached_df:

        # shuffle
        raw_df = pd.read_csv(DATA_PATH).sample(frac=1)

        # take subportion
        raw_df = raw_df[:row_lim]

        num_rows, _ = raw_df.shape
        
        print(f"# actual num rows taken = {num_rows}",)

        # preprocessed data
        data = DataManager(raw_df = raw_df, topic_limit = topic_limit, 
                            should_stem = False, should_clean = True, should_remove_stopwords = False, 
                            use_expanded_topics = use_expanded_topics,
                            )

    tidy_df = data.tidy_up(lang_to_vec)

    classification_stage='Super Topics'

    categorizer = Categorizer(df=tidy_df, none_ratio=none_ratio, 
                                classification_stage=classification_stage, 
                                chosen_topics=chosen_super_topics)

    OHE_df = categorizer.get_one_hot_encoded_df()

    # init a vectorizer that will convert string of words into numerical format
    vectorizer = TfidfVectorizer()

    # init class to split data
    splitter = DataSplitter(OHE_df, should_separate, DATA_PATH = DATA_PATH)

    # get subdivided datasets
    train, test, x_train, x_test, y_train, y_test = splitter.get_datasets(vectorizer)


    trainer = Trainer(classifier)

    trained_classifier = trainer.train(x_train, y_train)
    
    # init class of predictor based on classifier and list of chosen topics
    predictor = Predictor(trained_classifier, implement_rules = implement_rules, 
                            classification_stage = classification_stage,
                            top_topic_names = chosen_super_topics)
    
    # init class to constrcut confusion matrix
    cm_maker = ConfusionMatrix(categorizer.ranked_topic_names_with_none, 
                                # should_print = True
                                )

    # init class to compute scores
    scorer = Scorer(categorizer.get_topic_names_without_none(), categorizer.get_topic_counts_without_none(), row_lim, expt_num, none_ratio, 
                    should_print = True,
                    use_expanded_topics = use_expanded_topics, chosen_topics = chosen_super_topics
                    )

    # list of predictions
    train_pred_list = predictor.get_preds_list(x_train, text_df = train)
    test_pred_list = predictor.get_preds_list(x_test, text_df = test)

    # columns to compare pred and true labels
    train_pred_vs_true = predictor.get_pred_vs_true(train, train_pred_list)
    test_pred_vs_true = predictor.get_pred_vs_true(test, test_pred_list)

    multistage_classifier = MultiStageClassifier(expt_num, super_topics = chosen_super_topics)
    
    train_sorted_children_df = multistage_classifier.sort_children(train_pred_vs_true)
    test_sorted_children_df = multistage_classifier.sort_children(test_pred_vs_true)

    # probably need to evaluate -- i.e. score -- both stages of classification separately, as well as together.

    for super_topic in chosen_super_topics:

        # children_list = multistage_classifier.get_children_list(super_topic)

        y_train_df = multistage_classifier.get_numeric_df(train_sorted_children_df, super_topic)

        

        numeric_test_df = multistage_classifier.get_numeric_df(test_sorted_children_df, super_topic)




        print()

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

