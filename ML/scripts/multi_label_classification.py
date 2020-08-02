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
def time_and_reset(start_time):
    """
    Usage: 
    start_time = time_and_reset(start_time)
    """
    print('#',datetime.now() - start_time)
    return datetime.now()


def topic_limiter(row, permitted_topics):

    # this cell contains more topics than we might want
    old_passage_topics_string = row

    # compile list of this passage's topics, only including those which were top ranked
    new_passage_topics_list = [topic for topic in old_passage_topics_string.split() if topic in permitted_topics]
    
    # reconnect the topics in the list to form a string separated by spaces
    new_passage_topics_string = ' '.join(new_passage_topics_list)
    
    return new_passage_topics_string    
# **********************************************************************************************************
# settings

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
DATA_PATH = 'data/concat_english_prefix_hebrew.csv'

# classifiers = [BinaryRelevance(classifier=LinearSVC()),]
classifier = BinaryRelevance(classifier=LinearSVC())

# list of topics that you want to train and analyze
super_topics_list = [
    ['occurent', 'specifically-dependent-continuant','independent-continuant','generically-dependent-continuant'],
    ['generically-dependent-continuant', 'independent-continuant', 'occurent', 'quality', 'realizable-entity']
]

# how many topics to consider
topic_limit = None

# whether to use rule based logic after machine learning algorithm
implement_rules = False

# use cleaned stored dataframe, instead of always reprocessing it
use_cached_df = False

# which language(s) do you want to vectorize
# langs_to_vec = ['eng','heb','both']
lang_to_vec = 'eng'

# compute number of children for each node
get_ontology_counts = False

row_lim = 80000

# use number of nones that slightly greater than the most prevalent topic
none_ratio = 1.1

stages = [1,2]

# for expt_num, row_lim in enumerate(row_lims):
# for expt_num, stage in enumerate(stages):
for expt_num, super_topics in enumerate(super_topics_list):

    print(f'\n\n# expt #{expt_num} = {row_lim}')

    # super_topics = super_topics_list[expt_num]

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
                            # use_expanded_topics = use_expanded_topics,
                            )

    tidy_df = data.tidy_up(lang_to_vec)


    # produce additional columns with children of super topics: 
    multistage_classifier = MultiStageClassifier(expt_num, super_topics = super_topics)
    
    df = multistage_classifier.sort_children(tidy_df)

    # limit to only top ten topics in each super topic



    # categorizer = Categorizer(df=df, none_ratio=none_ratio, chosen_topics = super_topics)

    df['Super Topics'] = df['Super Topics'].apply(topic_limiter,args=(super_topics,))

    df = pd.concat([df, df['Super Topics'].str.get_dummies(sep=' ')], axis=1)


    family_topic_counts = {}
        
    OHE_dict = {}

    for super_topic in super_topics:

        child_col = f'True Children of {super_topic}'

        child_topics_lst = df[f'True Children of {super_topic}'].to_list()

        child_topics_str = ' '.join(child_topics_lst)

        all_topics_lst = child_topics_str.split()

        topic_counts = {}
        
        # loop thru all topic occurrences
        for topic in all_topics_lst:
        
            # increment if seen already
            if topic in topic_counts:

                topic_counts[topic] += 1
        
            # init if not seen yet
            else:

                topic_counts[topic] = 1

        counter = Counter(topic_counts)

        topic_counts = counter.most_common(1)

        top_topics = [topic_tuple[0] for topic_tuple in topic_counts]

        # family_topic_counts[super_topic] = permitted_topics

        df[child_col] = df[child_col].apply(topic_limiter,args=(top_topics,))

        df = pd.concat([df, df[child_col].str.get_dummies(sep=' ')], axis=1)

        # OHE_dict[super_topic] = true_topics

    topic_string_cols = [col for col in df.columns if "Topic" in col]

    for topic_string_col in topic_string_cols:
        
        df[topic_string_col] = df[topic_string_col].str.split()



    train, test = train_test_split(df, shuffle = True,test_size=0.30, random_state=42, )


    # select just the words of each passage
    train_text = train['passage_words']
    test_text = test['passage_words']

    vectorizer = TfidfVectorizer()

    # create document-term matrix, i.e. numerical version of passage text
    # Note: We only fit with training data, but NOT with testing data, because testing data should be "UNSEEN"
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)

    relevant_topics_dict = {}

    relevant_topics_dict['Super Topics'] = super_topics

    multistage_classifier = MultiStageClassifier(expt_num, super_topics = super_topics)
    
    for super_topic in super_topics:

        relevant_topics_dict[super_topic] = multistage_classifier.get_children_list(super_topic)

    # *******************************************************************
    # must change this for stage 2 of sub topics
    relevant_topics = relevant_topics_dict['Super Topics'] + ['None']
    # *******************************************************************

    relevant_cols = sorted([col for col in df.columns if col in relevant_topics])

    y_train = train[relevant_cols]
    y_test = test[relevant_cols]

    classifier.fit(x_train, y_train)

    pred_arrays_list = list(classifier.predict(x_test))

    true = y_test

    pred_labels_list = []
    
    for passage_index, passage_pred_array in enumerate(pred_arrays_list):
    
        if isinstance(passage_pred_array, scipy.sparse.csr.csr_matrix) or isinstance(passage_pred_array, np.int64) or isinstance(passage_pred_array, scipy.sparse.lil.lil_matrix):

            passage_pred_list = [passage_pred_array[0,i] for i in range(passage_pred_array.shape[1])]
    
        passage_labels = []
    
        for topic_index, pred_value in enumerate(passage_pred_list):
            
            topic_name = relevant_topics[topic_index]
            
            if pred_value != 0:

                passage_labels.append(topic_name)
    
        pred_labels_list.append(passage_labels)

    test['Pred Super Topics'] = pred_labels_list


    true_label_set_list = test['Super Topics'].tolist()
    pred_label_set_list = test['Pred Super Topics'].tolist()

    # check that we predicted label sets for the same number of passages as truly exist
    assert len(true_label_set_list) == len(pred_label_set_list)

    # how many passages in this set
    num_passages = len(true_label_set_list)

    # init 
    # e.g. 
    y_true = []
    y_pred = []

    for i in range(num_passages):

        # init, this parallel pair of lists is going to record what topics were (mis)matched
        # e.g. if there is one passage with 
        # true labels 'moses' and 'prayer', and 
        # pred labels 'moses' and 'abraham', then we would obtain
        # true_label_set = ['moses','prayer','None']
        # pred_label_set = ['moses','None','abraham']
        
        true_label_set = []
        pred_label_set = []
        
        try:
            true_label_set = true_label_set_list[i]
        except:
            pass
        
        try:
            pred_label_set = pred_label_set_list[i]
        except:
            pass

        # 0) NULL CASE --> No true or pred labels 
        # e.g. if there is one passage with no true labels and no pred labels 
        # true_label_set = ['None']
        # pred_label_set = ['None']

        if len(pred_label_set) == 0 and len(pred_label_set) == 0:
                y_true.append('None')
                y_pred.append('None')
    
        # 1) MATCH --> true label == pred label 
        # e.g. if there is one passage with 
        # true label 'moses', and 
        # pred label 'moses', then we would obtain
        # true_label_set = ['moses']
        # pred_label_set = ['moses']
        for true_label in true_label_set:
            if true_label in pred_label_set:
                y_true.append(true_label)
                y_pred.append(true_label)

        # 2) FALSE NEGATIVE --> true label was not predicted
        # e.g. if there is one passage with 
        # true label 'prayer', and no pred labels,
        # then we would obtain
        # true_label_set = ['prayer']
        # pred_label_set = ['None']
            else:
                y_true.append(true_label)
                y_pred.append("None")

        # 3) FALSE POSITIVE --> pred label was not true
        # e.g. if there is no true label, and the pred label is 'abraham', 
        # then we would obtain
        # true_label_set = ['None']
        # pred_label_set = ['abraham']
        for pred_label in pred_label_set:
            if pred_label not in true_label_set:
                y_true.append("None")
                y_pred.append(pred_label)
    
    y_actu = pd.Categorical(y_true, categories=relevant_topics)
    y_pred = pd.Categorical(y_pred, categories=relevant_topics)

    cm = pd.crosstab(y_actu, y_pred, rownames=['True'], colnames = ['Prediction'], dropna=False) 

    # normalize
    cm = cm.div(cm.sum(axis=1), axis=0).round(2)
    
    plt.figure()

    sns.heatmap(cm, annot=True)
    
    plt.savefig(f'super_topic_set_{expt_num}_row_lim_{row_lim}.png', bbox_inches='tight')



#     trainer = Trainer(classifier)

#     trained_classifier = trainer.train(x_train, y_train)
    
#     # init class of predictor based on classifier and list of chosen topics
#     predictor = Predictor(trained_classifier, implement_rules = implement_rules, 
#                             classification_stage = classification_stage,
#                             topic_names = chosen_super_topics)
    
#     # init class to constrcut confusion matrix
#     cm_maker = ConfusionMatrix(categorizer.ranked_topic_names_with_none, 
#                                 # should_print = True
#                                 )

#     # init class to compute scores
#     scorer = Scorer(categorizer.get_topic_names_without_none(), categorizer.get_topic_counts_without_none(), row_lim, expt_num, none_ratio, 
#                     should_print = True,
#                     use_expanded_topics = use_expanded_topics, chosen_topics = chosen_super_topics
#                     )

#     # list of predictions
#     train_pred_list = predictor.get_preds_list(x_train, text_df = train)
#     test_pred_list = predictor.get_preds_list(x_test, text_df = test)

#     # columns to compare pred and true labels
#     train_pred_vs_true = predictor.get_pred_vs_true(train, train_pred_list)
#     test_pred_vs_true = predictor.get_pred_vs_true(test, test_pred_list)

    
#     train_df = multistage_classifier.sort_children(train_pred_vs_true)
#     test_df = multistage_classifier.sort_children(test_pred_vs_true)

#     # probably need to evaluate -- i.e. score -- both stages of classification separately, as well as together.

#     for super_topic in chosen_super_topics:

#         # children_list = multistage_classifier.get_children_list(super_topic)

#         y_train_df = multistage_classifier.get_numeric_df(train_df, super_topic)
#         y_test_df = multistage_classifier.get_numeric_df(test_df, super_topic)

#         print()

#     # get confusion matrices
#     train_cm = cm_maker.get_cm_values(train_pred_vs_true)
#     test_cm = cm_maker.get_cm_values(test_pred_vs_true)

#     # check the worst performing examples to see what's going wrong
#     worst_train = cm_maker.check_worst(train_cm, train_pred_vs_true)
#     worst_test = cm_maker.check_worst(test_cm, test_pred_vs_true)

#     # get actual scores 
#     train_score_df = scorer.get_stats_df(train_cm, dataset = 'train')
#     test_score_df = scorer.get_stats_df(test_cm, dataset = 'test')

#     # save results
#     train_score_df.to_pickle(f'/persistent/Sefaria-Project/ML/data/train_score_df{none_ratio}.pkl')
#     test_score_df.to_pickle(f'/persistent/Sefaria-Project/ML/data/test_score_df_{none_ratio}.pkl')

# # compute and display time elapsed
# end_time = datetime.now()
# total_time = end_time - start_time
# print("\n# Total time taken:", total_time)
# print()

#     # RANK MODELS

