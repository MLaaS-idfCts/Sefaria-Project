import os
import glob
import pandas as pd
import warnings

from classes import DataManager, Predictor, Categorizer, Evaluator
from datetime import datetime
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain

start_time = datetime.now()

# ignore warnings regarding column assignment, 
# e.g. df['col1'] = list1 -- not 100% sure about this
pd.options.mode.chained_assignment = None  # default='warn'

# do not limit num of rows in df to display
pd.set_option('display.max_rows', None)

# disable future warnings
warnings.simplefilter(action='ignore', category = FutureWarning)

# width of column to dispaly in dataframe
pd.options.display.max_colwidth = 100

# ******************************************************************************************************

def refresh_scores():

    folders = ['images/scores/*','images/cm/*']

    for folder in folders:
            
        files = glob.glob(folder)

        for f in files:

            os.remove(f)

    with open("images/scores/scores_key.txt", "w") as file_object:

        file_object.write("scores_key")
    

def record_expt_specs(expt_num, vectorizer, classifier):

    with open("images/scores/scores_key.txt", "a") as file_object:

        file_object.write(f'\n')

        file_object.write(f'\nexpt_num: {expt_num}')

        file_object.write(f'\nvectorizer: {vectorizer}')

        file_object.write(f'\nclassifier: {classifier}')

        file_object.write(f'\nsuper_topics: {super_topics}')

# ******************************************************************************************************

DATA_PATH = 'data/concat_english_prefix_hebrew.csv'

classifier = LabelPowerset(classifier=LinearSVC())

vectorizer = CountVectorizer()

super_topics_list = [
    [
        'occurent', 'specifically-dependent-continuant',
        'independent-continuant', 'generically-dependent-continuant'
        ],
    [
        'generically-dependent-continuant', 'independent-continuant',
        'occurent', 'quality', 'realizable-entity'
        ]
]

lang_to_vec = 'eng' # ['eng','heb', 'both']

row_lim = 80000

max_children = 10

refresh_scores()

expt_num = 0

for super_topics in super_topics_list:

    expt_num += 1

    record_expt_specs(expt_num, vectorizer, classifier)

    data = DataManager(
        row_lim = row_lim, 
        data_path = DATA_PATH, 
        lang_to_vec = lang_to_vec, 
        should_stem = False, 
        super_topics = super_topics, 
        should_clean = True, 
        should_remove_stopwords = False,
        )

    data.prepare_dataframe()    

    categorizer = Categorizer(df = data.df, super_topics = super_topics)

    categorizer.sort_children(max_children = max_children)

    predictor = Predictor(
        df = categorizer.df,
        classifier = classifier,
        vectorizer = vectorizer,
        topic_lists = categorizer.topic_lists,
        super_topics = super_topics + ['entity'],
        )

    predictor.calc_results()

    evaluator = Evaluator(
        expt_num = expt_num, 
        data_sets = predictor.data_sets, 
        topic_lists = categorizer.topic_lists,
        super_topics = super_topics + ['entity'], 
        topic_counts = categorizer.topic_counts,
    ) 

    evaluator.calc_cm()

    evaluator.calc_scores()

    end_time = datetime.now()

    time_taken = end_time - start_time

    with open("images/scores/scores_key.txt", "a") as file_object:

        file_object.write(f'\ntime_taken: {time_taken}')

print()