import os
# from scripts.multi_label_classification import super_topics
import sys
import pickle
import django
import os.path

from os import path

sys.path.insert(1, '/persistent/Sefaria-Project/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'sefaria.settings'
django.setup()

from sefaria.model import *

import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from operator import itemgetter 
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
stemmer = SnowballStemmer('english')
np.seterr(divide='ignore', invalid='ignore')

class DataManager:

    def __init__(self, data_path, row_lim, 
                super_topics,
                lang_to_vec = 'eng',
                should_limit_nones = True, 
                should_stem = False, should_clean = True, keep_all_nones = False, 
                should_remove_stopwords = False, use_expanded_topics = False,
                ):
        
        self.data_path = data_path
        self.lang_to_vec = lang_to_vec
        self.row_lim = row_lim
        self.should_stem = should_stem
        self.super_topics = sorted(super_topics)
        self.should_clean = should_clean
        self.keep_all_nones = keep_all_nones
        self.should_limit_nones = should_limit_nones
        self.use_expanded_topics = use_expanded_topics
        self.should_remove_stopwords = should_remove_stopwords

    
    def get_ontology_counts_dict(self):

        df = self.preprocess_dataframe()

        # each item in list is a list of strings for one passage
        all_passage_node_lst = df['Expanded Topics'].tolist()
        
        all_nodes_str = ' '.join(all_passage_node_lst)
        
        all_nodes_lst = all_nodes_str.split()
        
        ontology_counts_dict = defaultdict(int)

        for node in all_nodes_lst:

            ontology_counts_dict[node] += 1
        
        # rank the entries by most frequently occurring first
        ontology_counts_ranked = {
            k:v for k, v in sorted(
                ontology_counts_dict.items(), 
                key=lambda item: item[1],
                reverse=True
                )
                }

        return ontology_counts_ranked


    def establish_dataframe(self):
        
        raw_df = pd.read_csv(self.data_path)            
        
        shuffled_df = raw_df.sample(frac=1,random_state=42)
        
        self.df = shuffled_df[:self.row_lim]
        
    
    def remove_junk_rows(self):
        
        self.establish_dataframe()
            
        # remove repeats
        self.df.drop_duplicates(inplace=True)
        
        # remove empty cells
        self.df.dropna(inplace=True)


    def clean(self):
        
        if self.should_clean:

            self.df['passage_text_english'] = self.df['passage_text_english'].str.lower()

            self.df['passage_text_english'] = self.df['passage_text_english'].apply(self.cleanHtml)

            self.df['passage_text_english'] = self.df['passage_text_english'].apply(self.cleanPunc)

            try:

                self.df['passage_text_hebrew_parsed'] = self.df['passage_text_hebrew_parsed'].apply(self.cleanHtml)

                self.df['passage_text_hebrew_parsed'] = self.df['passage_text_hebrew_parsed'].apply(self.cleanPunc)

            except:

                pass

            self.df['passage_text_english'] = self.df['passage_text_english'].apply(self.keepAlpha)
   
    
    def select_lang(self, lang_to_vec):
        
        
        if lang_to_vec == 'eng':
            
            self.df['passage_words'] = self.df['passage_text_english']

        if lang_to_vec == 'heb':

            self.df['passage_words'] = self.df['passage_text_hebrew_parsed']

        if lang_to_vec == 'both':

            self.df['passage_words'] = self.df['passage_text_english'] + ' ' + self.df['passage_text_hebrew_parsed'] 

    
    def remove_stopwords(self):
            
        if self.should_remove_stopwords:
            
            self.df['passage_text_english'] = self.df['passage_text_english'].apply(self.stopword_cleaner)
    
    
    def stem_words(self):
    
        if self.should_stem:
            
            self.df['passage_text_english'] = self.df['passage_text_english'].apply(self.stemmer)


    def select_columns(self):
    
        wanted_cols = ['passage_words','Topics','Expanded Topics']

        self.df = self.df[wanted_cols]

        self.df.rename(columns={'Topics': 'True Topics'}, inplace=True)

    
    def tidy_up(self):

        self.preprocess_dataframe()
            
        self.clean()
        
        self.remove_stopwords()
        
        self.stem_words()        


    def prepare_dataframe(self):
        
        self.tidy_up()

        self.select_lang(self.lang_to_vec)
                    
        self.select_columns()

        self.select_super_topics()


    def select_super_topics(self):

        new_col = 'True Super Topics'

        old_col = 'Expanded Topics'

        self.df[new_col] = self.df.pop(old_col).apply(
            TopicCounter().topic_limiter,
            args=(set(self.super_topics),))


    def remove_prefix(self, with_prefix):

        without_prefix = ' '.join([
            word[word.find('|') + 1:] 
            for word in with_prefix.split()
            ])
        
        return without_prefix


    def cleanHtml(self,sentence):

        cleanr = re.compile(r'<.*?>')

        cleantext = cleanr.sub(' ', sentence)

        return cleantext


    def cleanPunc(self,sentence): #function to clean the word of any punctuation or special characters
        
        # remove portions in parenthsesis or brackets
        cleaned = re.sub("([\(\[]).*?([\)\]])", " ", sentence)
        
        # remove punctuation characters
        cleaned = re.sub(r'[?|!|\'|"|#|.|,|)|(|\|/|:|-|—]',r' ',cleaned)
        
        # 
        cleaned = cleaned.strip()
        
        cleaned = cleaned.replace("\n"," ")
        
        # remove extra spaces
        cleaned = re.sub('\s+', ' ',cleaned)
        
        return cleaned


    def keepAlpha(self,sentence):

        # convert chars to acceptable format
        sentence = unidecode(sentence)
        
        # init
        alpha_sent = ""
        
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent


    def stopword_cleaner(self,sentence):
        stop_words = set(stopwords.words('english'))
        re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
        sentence = re_stop_words.sub(" ", sentence)
        return sentence


    def stemmer(self,sentence):
        stemmer = SnowballStemmer("english")
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence


    def preprocess_dataframe(self):

        self.remove_junk_rows()

        # use Ref as index instead of number
        self.df.set_index('Ref',drop=True, inplace=True)
    
        # make more descriptive name
        self.df.rename(columns={
            'En': 'passage_text_english',
            'He': 'passage_text_hebrew_unparsed',
            }, inplace=True)

        try:
            
            # remove prefixes from hebrew
            self.df['He_no_prefix'] = self.df.pop('He_prefixed').apply(self.remove_prefix)

            # make more descriptive name
            self.df.rename(columns={'He_no_prefix': 'passage_text_hebrew_parsed'}, inplace=True)

        except:
            
            pass



    def _get_ref_features(self,input_string):
        """
        Given a string, produce the substring that lies 
        after the last comma (if any) but 
        before the numbers at the end (if any).
        """
        result = input_string # init
        
        # get rid of everything before last comma
        last_comma = input_string.rfind(', ')
        if last_comma != -1:
            result = input_string[last_comma + 2:]

        # keep only letters and spaces
        result = ''.join(char for char in result if char.isalpha() or char == ' ')

        # remove single chars
        result = ' '.join( [w for w in result.split() if len(w)>1] )
        
        return result


    def _add_ref_features(self):
        df = self.stem_words()
        df['ref_features'] = df.Ref.apply(self._get_ref_features)
        return df


class Categorizer:

    def __init__(self, df, super_topics):

        self.df = df
        self.super_topics = sorted(super_topics)        
        self.topic_lists = {}
        self.topic_lists['Super Topics'] = self.super_topics
        

    def construct_children_list(self, super_topic):

        children_obj_lst = Topic.init(super_topic).topics_by_link_type_recursively()

        children_names_list = [child_obj.slug for child_obj in children_obj_lst]

        children_list_name = f"children_of_{super_topic}"

        path = f'data/{children_list_name}.pickle'

        with open(path, 'wb') as handle:
            
            pickle.dump(children_names_list, handle, protocol=3)

        return children_names_list



    def get_children(self, super_topic):

        self.super_topic = super_topic

        children_list_name = f"children_of_{super_topic}"

        path = f'data/{children_list_name}.pickle'

        if os.path.exists(path):

            with open(path, 'rb') as handle:

                children_names_list = pickle.load(handle)

        else:

            children_names_list = self.construct_children_list(super_topic)                

        children_names_list.remove(super_topic)
        
        return children_names_list 


    def make_child_column(self, super_topic):

            children = self.get_children(super_topic)
            
            self.df[f'True Children of {super_topic}'] = self.df['True Topics'].apply(
                                                                                TopicCounter().topic_limiter,
                                                                                args=[children]
                                                                                )


    def store_topic_counts(self, super_topic, max_children):

        topic_counts = TopicCounter().get_counts(self.df[f'True Children of {super_topic}'], max_children)
            
        self.topic_counts[super_topic] = topic_counts


    def get_child_names(self, super_topic, min_occurrences):

        topic_counts = self.topic_counts[super_topic]

        topic_names = [
            topic_count[0] 
            for topic_count in topic_counts 
            # if topic_count[1] >= min_occurrences
            ]

        self.topic_lists[f'Children of {super_topic}'] = topic_names

        return topic_names


    def limit_child_column(self, super_topic, topic_names):
        
        self.df[f'True Children of {super_topic}'] = self.df[f'True Children of {super_topic}'].apply(
            TopicCounter().topic_limiter, 
            args=[topic_names]
            )

        # print()


    def make_entity_column(self):

        self.make_child_column("entity")
        
        self.limit_child_column("entity", self.all_topics)
        
        self.topic_lists[f'Children of entity'] = sorted(list(self.all_topics))


    def make_child_columns(self, max_children, min_occurrences):

        for super_topic in self.super_topics:

            self.make_child_column(super_topic)

            self.store_topic_counts(super_topic, max_children)
            
            topic_names = self.get_child_names(super_topic, min_occurrences)

            # topic_names = get_popular_topics(topic_names, threshold)

            for topic_name in topic_names:
                
                self.all_topics.add(topic_name)

            self.limit_child_column(super_topic, topic_names)


    def count_all_topics(self):
        
        all_topic_counts = []

        for topic_counts in self.topic_counts.values():

            all_topic_counts.extend(topic_counts)

        all_topic_counts.sort(key=lambda x:x[1],reverse=True)

        self.topic_counts['entity'] = all_topic_counts

    
    def remove_unpopular_columns(self):

        all_cols = list(self.df.columns)
    
        # remove childless columns
        self.df = self.df.loc[:,self.df.any()]
    
        # remove unpopular columns
        populated_cols = list(self.df.columns)
        
        desolate_topics = [
            super_topic_group.split()[-1] 
            for super_topic_group in list(set(all_cols) - set(populated_cols))
            ]

        self.super_topics = list(set(self.super_topics) - set(desolate_topics))

        self.topic_lists['Super Topics'] = list(set(self.topic_lists['Super Topics']) - set(desolate_topics))

        # print()


    def remove_duplicated_columns(self):

        self.df = self.df.loc[:,~self.df.columns.duplicated()]

        # print()


    def sort_children(self, max_children, min_occurrences):

        self.topic_counts = {} # e.g. self.topic_counts[topic] = num_occurrences

        self.all_topics = set() # self.all_topics = {topic1, topic2, etc}

        self.make_child_columns(max_children, min_occurrences)

        self.remove_duplicated_columns()

        self.remove_unpopular_columns()

        self.make_entity_column()

        self.count_all_topics()


    def get_topic_names(self, ranked_topic_counts):
        
        return [topic_tuple[0] for topic_tuple in ranked_topic_counts]

    
    def get_topic_names_without_none(self):

        if getattr(self, "topic_names_without_none", None):

            pass

        else:

            topic_counts_without_none = self.get_topic_counts_without_none()

            topic_names_without_none = self.get_topic_names(topic_counts_without_none)

            self.topic_names_without_none = topic_names_without_none
        
        return self.topic_names_without_none


    def get_topic_counts_without_none(self):

        if getattr(self, "topic_counts_without_none", None):

            pass

        else:

            df = self.get_reduced_topics_df()

            # each item in this list is the string of topics for one passage
            all_passage_topics_lst = df[f'True {self.classification_stage}'].tolist()
            
            # huge string of all topics for all psasages
            all_topics_str = ' '.join(all_passage_topics_lst)
            
            # list of all topic instances
            all_topics_lst = all_topics_str.split()
            
            # init dict
            topic_counts_without_none_dict = defaultdict(int)
            
            # loop thru all topic occurrences
            for topic in all_topics_lst:
            
                topic_counts_without_none_dict[topic] += 1
            
            # rank the entries by most frequently occurring first
            topic_counts_without_none_dict = {
                                    k: v for k, v in sorted(topic_counts_without_none_dict.items(), 
                                    key=lambda item: item[1],
                                    reverse=True)
                                }
                
            # convert ranked dict into ranked list
            topic_counts_without_none_list = [(k, v) for k, v in topic_counts_without_none_dict.items()] 

            self.topic_counts_without_none = topic_counts_without_none_list

        return self.topic_counts_without_none


    def get_reduced_topics_df(self):

        if isinstance(getattr(self, "reduced_topics_df", None), pd.DataFrame):

            pass

        else:

            for col in self.df.columns:

                if "Children of" in col:

                    self.df[col] = self.df[col].apply(self.topic_selector)

        return self.df



    def limit_nones(self):

        if getattr(self, "limited_nones_df", None):

            pass

        else:
            
            df = self.get_reduced_topics_df()

            # place nones last
            df = df.sort_values(by=f'True {self.classification_stage}', ascending=False)

            # calc how many nones there are
            none_count = df.loc[df[f'True {self.classification_stage}'] == ""].shape[0]

            # num of occurrences of most popular topic 
            top_substantive_topic_count = self.get_topic_counts_without_none()[0][1]

            # init
            nones_to_drop = 0
            nones_to_keep = none_count
            
            if self.none_ratio == 'all':

                pass
                
            else:

                # compute num of nones to keep based upon ratio
                nones_to_keep = int(top_substantive_topic_count * self.none_ratio)

                # check there are more nones than the computed limit
                if nones_to_keep <= none_count:
                        
                    # calc how many nones to drop
                    nones_to_drop = none_count - nones_to_keep

                if nones_to_drop > 0:

                    # remove final excess 'none' rows
                    df = df.iloc[:-1 * nones_to_drop]

            # update list of topic counts
            self.ranked_topic_counts_with_none = self.get_topic_counts_without_none() + [('None',nones_to_keep)]
            
            # ensure 'None' is ordered acc to rank or occurrences, e.g. at the beginning if has most
            self.ranked_topic_counts_with_none.sort(key=lambda x:x[1],reverse=True)

            self.ranked_topic_names_with_none = self.get_topic_named_from_counts(self.ranked_topic_counts_with_none)

            self.limited_nones_df = df

        return self.limited_nones_df


class Predictor:    
    
    def __init__(self, classifier, vectorizer, df, super_topics, topic_lists):
        
        self.df = df
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.topic_lists = topic_lists
        self.super_topics = sorted(super_topics)


    def copy_all_topics(self):

        for data_set in ['train','test']:

            self.data_sets[data_set]['Pred Super Topics'] = self.data_sets[data_set]['True Super Topics']

            self.data_sets[data_set]['Pred Children of entity'] = self.data_sets[data_set]['True Children of entity']


    def address_super_topics(self):
                
        self.pred_super_topics()


    def calc_results(self):

        self.one_hot_encode()
            
        self.split_data()

        # self.address_super_topics()
        self.pred_super_topics()

        self.pred_sub_topics()

        self.tidy_data_sets()


    def train_test_split(self):

        self.data_sets = {}

        self.data_sets['train'], self.data_sets['test'] = train_test_split(self.df, shuffle = False, test_size=0.30)


    def store_text(self):

        self.text = {}

        for data_set in ['train','test']:

            self.text[data_set] = self.data_sets[data_set]['passage_words']


    def split_data(self):

        self.train_test_split()
        
        self.store_text()


    def pred_super_topics(self):
            
        self.topic_group = 'Super Topics'

        self.fit_and_pred()

            
    def pred_sub_topics(self):

        for super_topic in self.super_topics:

            self.super_topic = super_topic

            self.topic_group = f'Children of {super_topic}'

            self.fit_and_pred()


    def select_topics(self, topic_group):
        
        self.relevant_topics = self.topic_lists[topic_group]


    def get_wanted_cols(self):

        wanted_cols = [col for col in self.data_sets[self.data_set].columns if col in self.relevant_topics]
        return wanted_cols 
        # return  
        

    def set_x(self):

        self.x = {}

        self.x['train'] = self.vectorizer.fit_transform(self.text['train'])

        self.x['test'] = self.vectorizer.transform(self.text['test'])


    def set_y(self):

        self.y = {}

        for data_set in ['train','test']:

            self.data_set = data_set

            # wanted_cols = sorted(self.get_wanted_cols())

            # self.y[data_set] = self.data_sets[data_set][wanted_cols]

            df = self.data_sets[data_set][self.relevant_topics]

            df = df.loc[:,~df.columns.duplicated()]

            self.y[data_set] = df

        # print()


    def train_classifier(self):

        self.set_x()
        
        self.set_y()

        self.classifier.fit(self.x['train'], self.y['train'])


    def positive_result(self,pred_value):

        return pred_value != 0


    def child_of_pred(self, passage_index):

        # to test if the hierarchy algorithm will be viable at all, 
        # we feed the true super topics, as if they were predicted totally correct

        pred_super_topics = self.data_sets[self.data_set]['True Super Topics'][passage_index]
        # pred_super_topics = self.data_sets[self.data_set]['Pred Super Topics'][passage_index]

        topic_group_name = self.super_topic
        
        result = topic_group_name in pred_super_topics
        
        return result
    
    
    def is_super_topic(self):
    
        return self.topic_group in ['Super Topics', 'Children of entity']

    
    def topic_acceptable(self, passage_index):

        return self.is_super_topic() or self.child_of_pred(passage_index)
 
    
    def should_append(self, passage_index, pred_value):

        should_append = self.positive_result(pred_value) and self.topic_acceptable(passage_index) 
        
        return should_append


    def get_passage(self, passage_index):

        return self.data_sets[self.data_set]['passage_words'][passage_index] # for experimentation purposes


    def get_true_topics(self, passage_index):

        return self.data_sets[self.data_set].iloc[passage_index][f'True {self.topic_group}']


    def append_if_appropriate(self, topic_index, pred_value, passage_index):

        # self.get_passage(passage_index) # for expt purposes

        topic_name = self.relevant_topics[topic_index]

        # print(topic_index)

        if self.should_append(passage_index, pred_value):

            self.passage_labels.append(topic_name)
        

    def build_passage_labels(self, passage_index, pred_array):
        
        self.passage_labels = []

        passage_pred_list = [pred_array[0,i] for i in range(pred_array.shape[1])]
    
        for topic_index, pred_value in enumerate(passage_pred_list):

            # print(topic_index)
            
            self.append_if_appropriate(topic_index, pred_value, passage_index)


    def get_pred_labels_list(self):

        self.pred_labels_list = []

        for passage_index, pred_array in enumerate(self.pred_arrays):
    
            self.build_passage_labels(passage_index, pred_array)

            self.pred_labels_list.append(self.passage_labels)


    def make_predictions(self):

        input_data = self.x[self.data_set]

        predictions = self.classifier.predict(input_data)

        self.pred_arrays = list(predictions)


    def append_predictions(self):

        self.get_pred_labels_list() 

        self.data_sets[self.data_set][f'Pred {self.topic_group}'] = self.pred_labels_list


    def list_true_labels(self):

        true_cols = [col for col in self.data_sets[self.data_set].columns if 'True' in col]

        for true_col in true_cols:

            self.data_sets[self.data_set][true_col] = self.data_sets[self.data_set][true_col].str.split()


    def remove_irrelevant_columns(self):

        wanted_cols = [col for col in self.data_sets[self.data_set].columns if 'passage' in col or 'True' in col or 'Pred' in col]

        self.data_sets[self.data_set] = self.data_sets[self.data_set][wanted_cols]


    def tidy_data_sets(self):

        for data_set in ['train','test']:

            self.data_set = data_set

            self.list_true_labels()

            self.remove_irrelevant_columns()
            
    
    def predict(self):

        self.make_predictions()
        
        self.append_predictions()
        

    def fit_and_pred(self):

        self.relevant_topics = sorted(self.topic_lists[self.topic_group])

        self.train_classifier()

        for data_set in ['train','test']:

            self.data_set = data_set

            self.predict()

        # print()


    def remove_duplicated_columns(self):

        self.df = self.df.loc[:,~self.df.columns.duplicated()]


    def add_categorical_columns(self, col):

        self.df = pd.concat([self.df, self.df[col].str.get_dummies(sep=' ')], axis=1)


    def one_hot_encode(self):

        for stage in ['Super Topics', 'Children of entity']: 

            col = f'True {stage}'

            self.df = pd.concat([self.df, self.df[col].str.get_dummies(sep=' ')], axis=1)

        # print()


class ConfusionMatrix:

    def __init__(self, super_topic, cm_topics, expt_num):

        self.expt_num = expt_num
        self.cm_topics = cm_topics
        self.super_topic = super_topic


    def build_label_set_lists(self):

        cols = self.pred_vs_true[self.data_set].columns

        true_col = [col for col in cols if "True" in col][0]
        pred_col = [col for col in cols if "Pred" in col][0]

        self.true_label_set_list = self.pred_vs_true[self.data_set][true_col].tolist()
        self.pred_label_set_list = self.pred_vs_true[self.data_set][pred_col].tolist()

        assert len(self.true_label_set_list) == len(self.pred_label_set_list)

        self.num_passages = len(self.true_label_set_list)


    def get_cm_values(self):

        self.build_label_set_lists()

        y_true = []
        y_pred = []

        for i in range(self.num_passages):

            # init, this parallel pair of lists is going to record what topics were (mis)matched
            # e.g. if there is one passage with 
            # true labels 'moses' and 'prayer', and 
            # pred labels 'moses' and 'abraham', then we would obtain
            # true_label_set = ['moses','prayer','None']
            # pred_label_set = ['moses','None','abraham']
            
            true_label_set = []
            pred_label_set = []
            
            try:
                true_label_set = self.true_label_set_list[i]
            except:
                pass
            
            try:
                pred_label_set = self.pred_label_set_list[i]
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
        
        y_actu = pd.Categorical(y_true, categories=self.cm_topics)
        y_pred = pd.Categorical(y_pred, categories=self.cm_topics)

        cm = pd.crosstab(y_actu, y_pred, rownames=['True'], colnames = ['Prediction'], dropna=False) 

        fig = plt.figure()

        sns.heatmap(cm, annot=True,linewidths=1.0,cmap='summer')

        folder = 'images/cm'

        file_name = f'{self.expt_num}_{self.super_topic}_{self.data_set}.png'

        path = os.path.join(folder,file_name)

        plt.savefig(path, bbox_inches='tight')

        plt.close(fig)

        return cm


    def check_worst(self,cm, labels_df):
        """
        Given confusion matrix, calculate which topic statistically performs the worst, and produce those examples.
        """

        cm_norm = cm.div(cm.sum(axis=1), axis=0)

        if self.should_print:
        
            print(cm_norm.round(2))

        TP_rates = {}

        for topic in self.cm_topics:
        
            TP_rate = cm_norm.loc[topic,topic]
        
            TP_rates[topic] = TP_rate

        # TP rate for each topic, ordered worst to best, as a list of tuples
        ranked_TP_rates = sorted(TP_rates.items(), key=itemgetter(1))
        
        k = 1

        worst_topic = ranked_TP_rates[k][0]
        # worst_topic = ranked_TP_rates[0][0]

        # true, but not pred
        FN_df = labels_df[labels_df.true_topics.astype(str).str.contains(worst_topic) & ~labels_df.pred_topics.astype(str).str.contains(worst_topic)]
        # usage: print(FN_passages.to_string(index=False))
        FN_passages = FN_df[['passage_words']]

        FNs_with_keyword = FN_df[FN_df['passage_words'].str.contains(worst_topic)]


        # pred, but not true
        FP_df = labels_df[~labels_df.true_topics.astype(str).str.contains(worst_topic) & labels_df.pred_topics.astype(str).str.contains(worst_topic)]
        # usage: print(FP_passages.to_string(index=False))
        FP_passages = FP_df[['passage_words']]
        
        # easy_misses = FN_df[FN_df['passage_words'].str.contains(worst_topic)]

        return cm

class Scorer:


    def __init__(self, 
    # data_set, 
    expt_num, super_topic, topic_counts):

        # self.data_set = data_set
        self.expt_num = expt_num
        self.super_topic = super_topic
        self.topic_counts = topic_counts


    def get_precision(self, cm, topic, TP):

        FP = cm[topic].sum() - TP - cm.loc["None",topic]

        precision = TP/(TP + FP)

        return precision


    def get_recall(self, cm, topic, TP):

        FN = cm.loc[topic].sum() - TP

        recall = TP/(TP + FN)

        return recall
        

    def get_scores(self, cm):

        meaningful_topics = list(cm.columns)

        meaningful_topics.remove("None")

        precision_dict, recall_dict, f1score_dict = {}, {}, {}

        for topic in meaningful_topics:

            TP = cm.loc[topic,topic]

            recall = self.get_recall(cm, topic, TP)

            precision = self.get_precision(cm, topic, TP)
            
            f1score = 2 * (precision * recall)/(precision + recall)
            
            precision_dict[topic], recall_dict[topic], f1score_dict[topic] = precision, recall, f1score

        self.scores = {"Recall":recall_dict, 'F1score':f1score_dict, 'Precision':precision_dict} 


    def store_topic_occurrences(self):

        self.topic_stats_df = pd.DataFrame(self.topic_counts,columns=['Topic','Occurrences'])


    def store_total_occurrences(self):

        self.total_occurrences = sum(occurrences for topic, occurrences in self.topic_counts)

        assert self.total_occurrences == self.topic_stats_df.Occurrences.sum()


    def store_topic_proportions(self):

        self.topic_stats_df['Proportion'] = self.topic_stats_df.Occurrences/self.total_occurrences


    def store_scores(self):

        for score_type, scores_dict in self.scores.items():

            self.topic_stats_df[score_type] = self.topic_stats_df['Topic'].map(scores_dict)


    def calc_overall_stats(self):

        over_all_stats = {}

        for score_type in ['Recall','F1score','Precision']:

            over_all_stats[score_type] = (self.topic_stats_df[score_type] * self.topic_stats_df.Proportion).sum()

        over_all_stats['Topic'] = 'Overall'
        over_all_stats['Proportion'] = self.topic_stats_df.Proportion.sum() # ibid
        over_all_stats['Occurrences'] = self.topic_stats_df.Occurrences.sum() # exlcuding none occurrences

        self.topic_stats_df = self.topic_stats_df.append(over_all_stats, ignore_index=True)


    def get_stats_df(self, data_set, cm, super_topic):

        self.store_topic_occurrences()
        
        self.store_total_occurrences()

        self.store_topic_proportions()

        self.get_scores(cm)

        self.store_scores()

        self.calc_overall_stats()
        

class TopicCounter:

    @staticmethod
    def get_counts(series, max_topics):

        topic_set_lst = series.to_list()

        all_topics_str = ' '.join(topic_set_lst)

        all_topics_lst = all_topics_str.split()

        counter = Counter(all_topics_lst)

        topic_counts = counter.most_common(max_topics) 

        return topic_counts

    @staticmethod
    def topic_limiter(row, permitted_topics):

        # this cell contains more topics than we might want
        old_passage_topics_string = row

        # compile list of this passage's topics, only including those which were top ranked
        new_passage_topics_list = [topic for topic in old_passage_topics_string.split() if topic in permitted_topics]
        
        # reconnect the topics in the list to form a string separated by spaces
        new_passage_topics_string = ' '.join(new_passage_topics_list)
        
        return new_passage_topics_string    


class Evaluator:

    def __init__(
        self, 
        expt_num, 
        data_sets,
        topic_lists, 
        topic_counts, 
        super_topics, 
        ):

        self.expt_num = expt_num
        self.data_sets = data_sets
        self.topic_lists = topic_lists
        self.super_topics = sorted(super_topics)
        self.topic_counts = topic_counts


    def calc_cm(self):

        self.confusion_matrices = {}

        for super_topic in self.super_topics:

            cm_topics = self.topic_lists[f'Children of {super_topic}'] + ['None']

            cm_maker = ConfusionMatrix(
                cm_topics = cm_topics, 
                expt_num = self.expt_num,
                super_topic = super_topic, 
                )

            true_col = f'True Children of {super_topic}'
            pred_col = f'Pred Children of {super_topic}'

            cm_maker.pred_vs_true = {}

            self.confusion_matrices[super_topic] = {}

            for data_set in ['train','test']:

                cm_maker.data_set = data_set

                # record only columns of true vs pred
                cm_maker.pred_vs_true[cm_maker.data_set] = self.data_sets[cm_maker.data_set][[true_col,pred_col]]

                self.confusion_matrices[super_topic][cm_maker.data_set] = cm_maker.get_cm_values()


    def save_image(self, data, scoring_group, data_set):

        fig = plt.figure()

        score_chart = sns.barplot(x="Topic", y="F1score", data=data)

        score_chart.set_title(f'{scoring_group}\n{data_set}')

        plt.xticks(rotation=75, horizontalalignment='right')

        for p in score_chart.patches:

            score_chart.annotate(
                f"{round(p.get_height(),2)}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points'
                )

        # save image
        folder = 'images/scores'

        file_name = f'{self.expt_num}_{scoring_group}_{data_set}.png'

        path = os.path.join(folder,file_name)

        plt.savefig(path, bbox_inches='tight')

        plt.close(fig)
        

    def calc_scores(self):

        self.scores = {} # init

        self.overall_scores = {}

        for data_set in ['test','train']:

            self.overall_scores[data_set] = {}
            
            self.scores[data_set] = {}

        for super_topic in self.super_topics:
            
            scorer = Scorer(
                expt_num = self.expt_num, 
                super_topic = super_topic, 
                topic_counts = self.topic_counts[super_topic],
                )


            for data_set in ['test','train']:
                
                scorer.get_stats_df(
                    data_set = data_set,
                    super_topic = super_topic,
                    cm = self.confusion_matrices[super_topic][data_set],
                    )
                
                self.scores[data_set][super_topic] = scorer.topic_stats_df

                self.overall_scores[data_set][super_topic] = [
                    super_topic,
                    float(scorer.topic_stats_df.loc[scorer.topic_stats_df['Topic'] == 'Overall']['Occurrences']),
                    float(scorer.topic_stats_df.loc[scorer.topic_stats_df['Topic'] == 'Overall']['F1score'])
                    ]

                self.save_image(
                    data = scorer.topic_stats_df,
                    data_set = data_set,
                    scoring_group = super_topic, 
                    )

        for data_set in ['test','train']:
                        
            df = pd.DataFrame.from_dict(
                self.overall_scores[data_set], 
                orient='index',
                columns=['Topic', 'Occurrences', 'F1score']
                )

            families_df = df[df.Topic != 'entity']

            families_df['weighted_score'] = families_df.Occurrences * families_df.F1score

            total_occurrences = families_df.Occurrences.sum()

            avg_score = families_df.weighted_score.sum()/total_occurrences

            df.loc['hierarchy'] = ['hierarchy', total_occurrences, avg_score]

            self.save_image(
                data = df,  
                data_set = data_set,
                scoring_group = 'Overall', 
                )
