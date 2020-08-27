import os
import glob
# from scripts.table_of_contents import div_laws
import pickle
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
pd.options.display.max_colwidth = 40

# ******************************************************************************************************

def time_keeper(start_time):

    current_time = datetime.now()

    time_taken = current_time - start_time

    print(time_taken)

    return current_time


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

topic_groups_list = {}

div_laws_options = ['laws_united','laws_divided']

for i, div_laws in enumerate(div_laws_options):

    with open(f'data/topic_groups_{div_laws_options[i]}.pickle', 'rb') as handle:

        topic_groups_list[i] = pickle.load(handle)

super_topics_list = [
    # ['entity'], # 
    # topic_groups_list[0], # laws_united 
    topic_groups_list[1], # laws_divided
    # ['occurent', 'specifically-dependent-continuant','independent-continuant', 'generically-dependent-continuant'],
    # ['generically-dependent-continuant', 'independent-continuant','occurent', 'quality', 'realizable-entity']
]

super_topics = super_topics_list[0]

lang_to_vec = 'eng' # ['eng','heb', 'both']

# row_lim = 100
# row_lim = 500
# row_lim = 1000
row_lim = 5000
# row_lim = 10000
# row_lim = 20000
# row_lim = 40000
# row_lim = 80000

max_children = 1
# max_children = 2
# max_children = 5
# max_children = 10
# max_children = 100

min_occurrences = 1
# min_occurrences = 5
# min_occurrences = 20
# min_occurrences = 50
# min_occurrences = 100

family_pred_options = [True, False]

true_family_given = False

use_rules_options = [False, True]

refresh_scores()

expt_num = 0

# use_rules = True
use_rules = False

# use_ML = False
use_ML = True

if True:
# for i, use_rules in enumerate(use_rules_options):
# for i, true_family_given in enumerate(family_pred_options):
# for i, super_topics in enumerate(super_topics_list):

    expt_num += 1

    record_expt_specs(expt_num, vectorizer, classifier)

    start_time = time_keeper(start_time)

    print('DataManager')
    
    data = DataManager(
        row_lim = row_lim, 
        data_path = DATA_PATH, 
        lang_to_vec = lang_to_vec, 
        should_stem = False, 
        super_topics = sorted(super_topics), # very impt to preserve order, e.g. alphabetical
        should_clean = True, 
        should_remove_stopwords = False,
        )

    data.prepare_dataframe()    

    start_time = time_keeper(start_time)

    print('Categorizer')
    
    categorizer = Categorizer(
        df = data.df, 
        super_topics = data.super_topics
        )

    categorizer.sort_children(
        max_children = max_children, 
        min_occurrences = min_occurrences
        )

    start_time = time_keeper(start_time)
    
    print('Predictor')
    
    predictor = Predictor(
        df = categorizer.df,
        use_rules = use_rules,
        use_ML = use_ML,
        classifier = classifier,
        vectorizer = vectorizer,
        true_family_given = true_family_given,
        topic_lists = categorizer.topic_lists,
        super_topics = categorizer.super_topics + ['entity'],
        )

    predictor.calc_results()

    start_time = time_keeper(start_time)
    
    print('Evaluator')
    
    evaluator = Evaluator(
        expt_num = expt_num, 
        data_sets = predictor.data_sets, 
        topic_lists = categorizer.topic_lists,
        super_topics = predictor.super_topics, 
        true_family_given = true_family_given,
        topic_counts = categorizer.topic_counts,
    ) 

    evaluator.calc_cm()

    evaluator.calc_scores()
    
    # evaluator.check_worst()

    start_time = time_keeper(start_time)

    # with open("images/scores/scores_key.txt", "a") as file_object:

    #     file_object.write(f'\ntime_taken: {time_taken}')

    print()

print()