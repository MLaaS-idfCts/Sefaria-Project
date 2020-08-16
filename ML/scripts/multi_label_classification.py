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
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

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
    LabelPowerset(classifier=LinearSVC()),
    LabelPowerset(classifier=DecisionTreeClassifier()),
    LabelPowerset(classifier=RandomForestClassifier(n_estimators = 10)),
    # LabelPowerset(classifier=GradientBoostingClassifier(n_estimators = 10)),
]

vectorizers = [
    CountVectorizer(),
    TfidfVectorizer(),
    HashingVectorizer(),
]

super_topics_list = [
    ['entity'],
    ['generically-dependent-continuant'],
    # ['occurent', 'specifically-dependent-continuant','independent-continuant','generically-dependent-continuant'],
    # ['generically-dependent-continuant', 'independent-continuant', 'occurent', 'quality', 'realizable-entity']
]

langs = ['eng','heb','both']
lang_to_vec = 'eng'

row_lim = 18000

expt_num = 0


folder = 'images/scores/*'
files = glob.glob(folder)
for f in files:
    os.remove(f)


with open("images/scores/scores_key.txt", "w") as file_object:
    file_object.write("scores_key")

for vectorizer in vectorizers:

    for classifier in classifiers:
            
        for super_topics in super_topics_list:

            expt_num += 1

            with open("images/scores/scores_key.txt", "a") as file_object:
                file_object.write(f'\n')
                file_object.write(f'\nexpt_num: {expt_num}')
                file_object.write(f'\nvectorizer: {vectorizer}')
                file_object.write(f'\nclassifier: {classifier}')
                file_object.write(f'\nsuper_topics: {super_topics}')

            data = DataManager(
                data_path = DATA_PATH, row_lim = row_lim, 
                super_topics = super_topics, lang_to_vec = lang_to_vec, 
                should_stem = False, should_clean = True, should_remove_stopwords = False
                )

            data.prepare_dataframe()    

            categorizer = Categorizer(df = data.df, super_topics = super_topics)

            max_children = None

            if super_topics == ['entity']:

                max_children = 30

            else:

                max_children = 10

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

            with open("images/scores/scores_key.txt", "a") as file_object:
                file_object.write(f'\ntime_taken: {time_taken}')

print()
