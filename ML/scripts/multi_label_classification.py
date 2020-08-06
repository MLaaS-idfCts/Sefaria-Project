
import pandas as pd
import warnings

from classes import DataManager, Predictor, Categorizer, Evaluator
from datetime import datetime
from sklearn.svm import SVC, LinearSVC
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from sklearn.feature_extraction.text import TfidfVectorizer

start_time = datetime.now()

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

# ignore warnings regarding column assignment, e.g. df['col1'] = list1 -- not 100% sure about this
pd.options.mode.chained_assignment = None  # default='warn'

# do not limit num of rows in df to display
pd.set_option('display.max_rows', None)

# disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# width of column to dispaly in dataframe
pd.options.display.max_colwidth = 150


DATA_PATH = 'data/concat_english_prefix_hebrew.csv'

classifier = BinaryRelevance(classifier=LinearSVC())

vectorizer = TfidfVectorizer()

super_topics_list = [
    ['occurent', 'specifically-dependent-continuant','independent-continuant','generically-dependent-continuant'],
    ['generically-dependent-continuant', 'independent-continuant', 'occurent', 'quality', 'realizable-entity']
]

# which language(s) do you want to vectorize; langs_to_vec = ['eng','heb','both']
lang_to_vec = 'eng'

row_lim = None

for expt_num, super_topics in enumerate(super_topics_list):

    for discriminate_families in [True, False]:

        print(f'# expt_num #{expt_num}\n# {len(super_topics)} super_topics: {super_topics}')

        data = DataManager(data_path = DATA_PATH, row_lim = row_lim, 
                            super_topics = super_topics, lang_to_vec = lang_to_vec, 
                            should_stem = False, should_clean = True, 
                            should_remove_stopwords = False)

        data.prepare_dataframe()    

        categorizer = Categorizer(df = data.df, super_topics=super_topics)

        categorizer.sort_children(max_children = 10)

        predictor = Predictor(classifier = classifier, vectorizer = vectorizer, df = categorizer.df,
                                super_topics = super_topics, topic_lists = categorizer.topic_lists)

        predictor.discriminate_families = discriminate_families

        predictor.calc_results()

        evaluator = Evaluator(
            expt_num = expt_num, 
            data_sets = predictor.data_sets, 
            topic_lists = categorizer.topic_lists,
            super_topics = super_topics, 
            topic_counts = categorizer.topic_counts,
        ) 

        evaluator.discriminate_families = discriminate_families

        evaluator.calc_cm()

        evaluator.calc_scores()
        
        # evaluator.plot_results()
    print()


