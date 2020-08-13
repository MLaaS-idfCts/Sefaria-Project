
from inspect import classify_class_attrs
from math import exp
import pandas as pd
import warnings

from classes import DataManager, Predictor, Categorizer, Evaluator
from datetime import datetime
from sklearn.svm import SVC, LinearSVC
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from sklearn.feature_extraction.text import TfidfVectorizer

start_time = datetime.now()

# ignore warnings regarding column assignment, e.g. df['col1'] = list1 -- not 100% sure about this
pd.options.mode.chained_assignment = None  # default='warn'

# do not limit num of rows in df to display
pd.set_option('display.max_rows', None)

# disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# width of column to dispaly in dataframe
pd.options.display.max_colwidth = 150

DATA_PATH = 'data/concat_english_prefix_hebrew.csv'

classifiers = [
    BinaryRelevance(classifier=LinearSVC()),
    # BinaryRelevance(classifier=SVC()),

    ClassifierChain(classifier=LinearSVC()),
    # ClassifierChain(classifier=SVC()),

    LabelPowerset(classifier=LinearSVC()),
    # LabelPowerset(classifier=SVC()),
]

vectorizer = TfidfVectorizer()

super_topics_list = [
    ['entity'],
    ['generically-dependent-continuant'],
    # ['occurent', 'specifically-dependent-continuant','independent-continuant','generically-dependent-continuant'],
    # ['generically-dependent-continuant', 'independent-continuant', 'occurent', 'quality', 'realizable-entity']
]

langs = ['eng','heb','both']
lang_to_vec = 'eng'

row_lim = 2000

expt_num = 0

for classifier in classifiers:
        
    for super_topics in super_topics_list:

        expt_num += 1

        data = DataManager(
            data_path = DATA_PATH, row_lim = row_lim, 
            super_topics = super_topics, lang_to_vec = lang_to_vec, 
            should_stem = False, should_clean = True, should_remove_stopwords = False
            )

        data.prepare_dataframe()    

        categorizer = Categorizer(df = data.df, super_topics = super_topics)

        max_children = 10
        # max_children = int(5/len(super_topics))

        categorizer.sort_children(max_children = max_children)

        predictor = Predictor(
            classifier = classifier, vectorizer = vectorizer, 
            df = categorizer.df, super_topics = super_topics, 
            topic_lists = categorizer.topic_lists
            )

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

        end_time = datetime.now()

        time_taken = end_time - start_time

        print('expt_num:',expt_num)
        print('classifier:',classifier)
        print('super_topics:',super_topics)
        print('time_taken',time_taken)

print()
