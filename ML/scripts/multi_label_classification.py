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
from classes import DataManager, ConfusionMatrix, Predictor, DataSplitter, \
                    Scorer, Trainer, Categorizer, MultiStageClassifier, \
                    Evaluator
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
pd.options.display.max_colwidth = 150


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

row_lim = 10000

for expt_num, super_topics in enumerate(super_topics_list):

    for only_pred_super in [True, False]:

        print(f'# expt_num #{expt_num}\n# {len(super_topics)} super_topics: {super_topics}')

        data = DataManager(data_path = DATA_PATH, row_lim = row_lim, 
                            super_topics = super_topics, lang_to_vec = lang_to_vec, 
                            should_stem = False, should_clean = True, 
                            should_remove_stopwords = False)

        data.prepare_dataframe()    

        categorizer = Categorizer(df = data.df, super_topics=super_topics)

        categorizer.sort_children(max_children = 3)

        predictor = Predictor(classifier = classifier, vectorizer = vectorizer, df = categorizer.df,
                                super_topics = super_topics, topic_lists = categorizer.topic_lists)

        predictor.calc_results()

        evaluator = Evaluator(
            expt_num = expt_num, 
            data_sets = predictor.data_sets, 
            topic_lists = categorizer.topic_lists,
            super_topics = super_topics, 
            topic_counts = categorizer.topic_counts,
        ) 

        evaluator.calc_cm()

        evaluator.calc_scores()
        
        evaluator.plot_results()

    print()


