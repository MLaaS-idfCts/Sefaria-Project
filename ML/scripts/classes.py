import os
import sys
import csv
import pickle
import django
import os.path

from os import path

sys.path.insert(1, '/persistent/Sefaria-Project/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'sefaria.settings'
django.setup()

from tqdm import tqdm
from sefaria.model import *
# from sefaria.model.topic import topics_by_link_type_recursively
from sefaria.system.database import db


import re
import nltk
import scipy
import numpy as np
import pandas as pd
import random
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection

from bs4 import BeautifulSoup
from tqdm import tqdm
from random import shuffle
from string import printable
from operator import itemgetter 
from datetime import datetime
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from collections import Counter
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from IPython.display import HTML
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
stemmer = SnowballStemmer('english')
np.seterr(divide='ignore', invalid='ignore')

class DataManager:

    def __init__(self, data_path, row_lim, topic_limit, 
                super_topics,
                should_limit_nones = False, 
                should_stem = False, should_clean = True, keep_all_nones = False, 
                should_remove_stopwords = False, use_expanded_topics = False,
                ):
        
        self.data_path = data_path
        self.row_lim = row_lim
        self.should_stem = should_stem
        self.topic_limit = topic_limit
        self.super_topics = super_topics
        self.should_clean = should_clean
        self.keep_all_nones = keep_all_nones
        self.should_limit_nones = should_limit_nones
        self.use_expanded_topics = use_expanded_topics
        self.should_remove_stopwords = should_remove_stopwords

    
    def get_ontology_counts_dict(self):

        df = self.preprocess_dataframe()

        # each item in list is a string of lists for one passage
        all_passage_node_lst = df['Expanded Topics'].tolist()
        
        # huge string of all topic for all psasages
        all_nodes_str = ' '.join(all_passage_node_lst)
        
        # list of all topic instances
        all_nodes_lst = all_nodes_str.split()
        
        # init dict
        ontology_counts_dict = {}
        
        # loop thru all topic occurrences
        for node in all_nodes_lst:
        
            # increment if seen already
            if node in ontology_counts_dict:
                ontology_counts_dict[node] += 1
        
            # init if not seen yet
            else:
                ontology_counts_dict[node] = 1
        
        # rank the entries by most frequently occurring first
        ontology_counts_dict = {
                                k: v for k, v in sorted(ontology_counts_dict.items(), 
                                key=lambda item: item[1],
                                reverse=True)
                            }

        return ontology_counts_dict



    def establish_dataframe(self):
        
        if isinstance(getattr(self, "dataframe", None), pd.DataFrame):
            
            return self.dataframe

        else:
            
            raw_df = pd.read_csv(self.data_path)            
            shuffled_df = raw_df.sample(frac=1,random_state=42)
            df = shuffled_df[:self.row_lim]
            num_rows, _ = df.shape
            print(f"# actual num rows taken = {num_rows}",)
        return df
    
    
    def remove_junk_rows(self):
        
        if isinstance(getattr(self, "without_junk_rows", None), pd.DataFrame):
            
            return self.without_junk_rows

        else:
            
            df = self.establish_dataframe()
            
            # remove repeats
            df = df.drop_duplicates()
            
            # remove empty cells
            df = df.dropna()
            
            # store as attribute
            self.without_junk_rows = df

        return df




    def tidy_up(self, lang_to_vec):

        if self.should_clean:

            df = self.preprocess_dataframe()

            df['passage_text_english'] = df['passage_text_english'].str.lower()
            df['passage_text_english'] = df['passage_text_english'].apply(self.cleanHtml)
            df['passage_text_english'] = df['passage_text_english'].apply(self.cleanPunc)

            try:
                df['passage_text_hebrew_parsed'] = df['passage_text_hebrew_parsed'].apply(self.cleanHtml)
                df['passage_text_hebrew_parsed'] = df['passage_text_hebrew_parsed'].apply(self.cleanPunc)
            except:
                pass

            df['passage_text_english'] = df['passage_text_english'].apply(self.keepAlpha)
    
        if self.should_remove_stopwords:
            df['passage_text_english'] = df['passage_text_english'].apply(self.stopword_cleaner)
    
        if self.should_stem:
            df['passage_text_english'] = df['passage_text_english'].apply(self.stemmer)
            
        if lang_to_vec == 'eng':
            df['passage_words'] = df['passage_text_english']

        if lang_to_vec == 'heb':
            df['passage_words'] = df['passage_text_hebrew_parsed']

        if lang_to_vec == 'both':
            df['passage_words'] = df['passage_text_english'] + ' ' + df['passage_text_hebrew_parsed'] 

        wanted_cols = ['passage_words','Topics','Expanded Topics']

        df = df[wanted_cols]

        df.rename(columns={'Topics': 'True Topics'}, inplace=True)

        topic_counter = TopicCounter()

        df['True Super Topics'] = df.pop('Expanded Topics').apply(
                                                            topic_counter.topic_limiter,
                                                            args=(self.super_topics,)
                                                            )

        return df


    def remove_prefix(self, row):

        with_prefix_str = row

        with_prefix_lst = with_prefix_str.split()

        without_prefix_lst = []

        for word in with_prefix_lst:

            word_no_prefix = word[word.find('|') + 1:]

            without_prefix_lst.append(word_no_prefix)

        without_prefix_str = ' '.join(without_prefix_lst)
        
        return without_prefix_str


    def cleanHtml(self,sentence):

        try:
            soup = BeautifulSoup(sentence,features="lxml")
            cleantext = soup.get_text()

        except:
            cleanr = re.compile(r'<.*?>')
            cleantext = cleanr.sub('', sentence)

        return cleantext


    def cleanPunc(self,sentence): #function to clean the word of any punctuation or special characters
        
        # remove portions in parenthsesis or brackets
        cleaned = re.sub("([\(\[]).*?([\)\]])", "", sentence)
        
        # remove punctuation characters
        cleaned = re.sub(r'[?|!|\'|"|#|.|,|)|(|\|/|:|-|—]',r' ',cleaned)
        
        # 
        cleaned = cleaned.strip()
        
        cleaned = cleaned.replace("\n"," ")
        
        # remove extra spaces
        cleaned = re.sub(' +', ' ',cleaned)
        
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

        if isinstance(getattr(self, "preprocessed_dataframe", None),pd.DataFrame):

            return self.preprocessed_dataframe
        
        else:
            df = self.remove_junk_rows()
        
            # use Ref as index instead of number
            df = df.set_index('Ref',drop=True)
        
            # df = df.drop(columns=['Topics'])

            # make more descriptive name
            df = df.rename(columns={
                'En': 'passage_text_english',
                'He': 'passage_text_hebrew_unparsed',
                })

            try:
                # remove prefixes from hebrew
                df['He_no_prefix'] = df.pop('He_prefixed').apply(self.remove_prefix)

                # make more descriptive name
                df = df.rename(columns={
                    'He_no_prefix': 'passage_text_hebrew_parsed',
                    })

            except:
                pass
    
            self.preprocessed_dataframe = df

        return df


    def show_topic_counts(self):
        df = self.preprocess_dataframe()
        categories = list(df.columns.values)
        categories = categories[1:]
        sns.set(font_scale = 2)
        figure = plt.figure(figsize=(15,8))

        ax= sns.barplot(categories, df.iloc[:,1:].sum().values)

        plt.title("Passages in each topic", fontsize=24)
        plt.ylabel('Number of Passages', fontsize=18)
        plt.xlabel('Passage Type ', fontsize=18)

        #adding the text labels
        rects = ax.patches
        labels = df.iloc[:,1:].sum().values
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, height + 0, label, ha='center', va='bottom', fontsize=18)

        plt.xticks(rotation=90)
        # plt.imshow()
        # plt.show()
        return ax


    def show_multiple_labels(self):
        rowSums = df.iloc[:,2:].sum(axis=1)
        multiLabel_counts = rowSums.value_counts()
        multiLabel_counts.sort_index(inplace=True)
        multiLabel_counts = multiLabel_counts.iloc[:]

        sns.set(font_scale = 2)
        plt.figure(figsize=(15,8))

        ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)

        plt.title("passages having multiple labels ")
        plt.ylabel('Number of passages', fontsize=18)
        plt.xlabel('Number of labels', fontsize=18)
        #adding the text labels
        rects = ax.patches
        labels = multiLabel_counts.values
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, height + 0, label, ha='center', va='bottom')
        plt.show()


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
        self.super_topics = super_topics
        self.topic_lists = {}
        self.topic_lists['Super Topics'] = self.super_topics
        

    def get_children_list(self, super_topic):

        self.super_topic = super_topic

        children_list_name = f"children_of_{super_topic}"

        path = f'data/{children_list_name}.pickle'

        if os.path.exists(path):

            with open(path, 'rb') as handle:

                children_names_list = pickle.load(handle)

        else:
                
            super_topic_obj = Topic.init(super_topic)

            children_obj_lst = super_topic_obj.topics_by_link_type_recursively()

            children_names_list = []

            for child_obj in children_obj_lst:

                child_name = child_obj.slug

                children_names_list.append(child_name)

            with open(path, 'wb') as handle:
                
                pickle.dump(children_names_list, handle, protocol=3)

        children_names_list.remove(super_topic)
        
        return children_names_list 





    def get_numeric_df(self, df, super_topic):

        children_topics = self.get_children_list(super_topic)

        topic_col = [col for col in df.columns if super_topic in col][0]
        
        cols = ['passage_words',topic_col]

        df = df[cols]

        df.rename(columns={topic_col: 'Topics'}, inplace=True)

        categorizer = Categorizer(df=df, classification_stage='Topics', chosen_topics=children_topics)

        one_hot_encoded_df = categorizer.get_one_hot_encoded_df()

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        OHE_topics = one_hot_encoded_df.select_dtypes(include=numerics)

        return OHE_topics


    def sort_children(self, max_children):

        # categorizer = Categorizer()

        topic_counter = TopicCounter()

        for super_topic in self.super_topics:

            children_names_lst = self.get_children_list(super_topic)

            self.df[f'True Children of {super_topic}'] = self.df['True Topics'].apply(
                                                                                topic_counter.topic_limiter,
                                                                                args=[children_names_lst]
                                                                                )

            topic_counts = topic_counter.get_counts(self.df[f'True Children of {super_topic}'], max_children)

            topic_names = [topic_count[0] for topic_count in topic_counts]

            self.topic_lists[f'Children of {super_topic}'] = topic_names

            self.df[f'True Children of {super_topic}'] = self.df[f'True Children of {super_topic}'].apply(
                                                                                topic_counter.topic_limiter,
                                                                                args=[topic_names]
                                                                                )


        return self.df


    def get_topic_names(self, ranked_topic_counts):

        return  [topic_tuple[0] for topic_tuple in ranked_topic_counts]

    
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
            topic_counts_without_none_dict = {}
            
            # loop thru all topic occurrences
            for topic in all_topics_lst:
            
                # increment if seen already
                if topic in topic_counts_without_none_dict:

                    topic_counts_without_none_dict[topic] += 1
            
                # init if not seen yet
                else:

                    topic_counts_without_none_dict[topic] = 1
            
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

    
    def get_one_hot_encoded_df(self):

        if getattr(self,'one_hot_encoded_df',None):

            pass

        else:

            df = self.limit_nones()

            # one hot encode each topic
            df = pd.concat([df, df[f'True {self.classification_stage}'].str.get_dummies(sep=' ')], axis=1)

            # make topic string into list
            df[f'True {self.classification_stage}'] = df[f'True {self.classification_stage}'].str.split()

            self.one_hot_encoded_df = df

        return self.one_hot_encoded_df


class DataSplitter:

    def __init__(self, data_df, should_separate, DATA_PATH):
    
        self.data_df = data_df
        self.DATA_PATH = DATA_PATH
        self.should_separate = should_separate
        

    def get_datasets(self, vectorizer):

        # arrange in order of index for passage
        df = self.data_df.sort_values(by='Ref')

        print('\n# should_separate =',self.should_separate)

        if self.should_separate:
            
            if 'concat' in self.DATA_PATH:

                all_refs_list = list(df.index)

            if 'multi_version' in self.DATA_PATH:

                all_refs_list = [ref_vsn[:ref_vsn.find(' -')] for ref_vsn in list(df.index)]

            refs_set = set(all_refs_list)
            refs_list = list(refs_set)

            random.seed(4)
            random.shuffle(refs_list)

            test_portion = 0.3
            
            num_refs = len(refs_list)
            
            num_test_rows = int(test_portion * num_refs)

            num_train_rows = num_refs - num_test_rows

            actual_test_portion = num_test_rows/num_refs

            train_refs = refs_list[:num_train_rows + 1]
            test_refs = refs_list[num_train_rows + 1:]

            if 'multi_version' in self.DATA_PATH:

                df['Ref_with_version'] = df.index
                # df['Ref_only'] = df.Ref_with_version.str[:df.Ref_with_version.str.find(' -- ')]
                with_version = df.Ref_with_version.str
                parsed_list = with_version.split(' -- ')
                df['Ref_only'] = parsed_list.str[0]
                # print()

            if 'concat' in self.DATA_PATH:

                df['Ref_only'] = df.index


            train = df[df['Ref_only'].isin(train_refs)]
            test = df[df['Ref_only'].isin(test_refs)]


            print(f'# actual test portion = {actual_test_portion}')


        if not self.should_separate:
            
            train, test = train_test_split(
                        df, 
                        shuffle = True,
                        # shuffle = False,
                        test_size=0.30, 
                        random_state=42, 
                    )

        # select just the words of each passage
        train_text = train['passage_words']
        test_text = test['passage_words']

        # create document-term matrix, i.e. numerical version of passage text
        # Note: We only fit with training data, but NOT with testing data, because testing data should be "UNSEEN"
        x_train = vectorizer.fit_transform(train_text)
        x_test = vectorizer.transform(test_text)
        start_time = datetime.now()

        # topics columns, with 0/1 indicating if the topic of this column relates to that row's passage
        
        # all_cols = list(self.data_df.columns)
        
        # cols_to_keep = all_cols[all_cols.index('true_topics') + 1:-1]
        y_train = train._get_numeric_data()
        y_test = test._get_numeric_data()
        # y_train = train[cols_to_keep]
        # y_test = test[cols_to_keep]

        return train, test, x_train, x_test, y_train, y_test





class Predictor:
    
    
    def __init__(self, classifier, vectorizer, df, super_topics, topic_lists):
        
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.df = df
        self.super_topics = super_topics
        self.topic_lists = topic_lists


    def split_data(self):

        self.one_hot_encode()
        
        self.train_set, self.test_set = train_test_split(self.df, shuffle = False, test_size=0.30)

        self.train_text = self.train_set['passage_words']
        self.test_text = self.test_set['passage_words']


    def select_topics(self, topic_group):
        
        self.relevant_topics = self.topic_lists[topic_group]


    def train_classifier(self):

        # pass

        self.x_train = self.vectorizer.fit_transform(self.train_text)
        self.x_test = self.vectorizer.transform(self.test_text)

        self.y_train = self.train_set[self.relevant_topics]
        self.y_test = self.test_set[self.relevant_topics]

        self.classifier.fit(self.x_train, self.y_train)


    def get_pred_labels_list(self, data_set, pred_arrays_list, topic_group):

        pred_labels_list = []

        for passage_index, pred_array in enumerate(pred_arrays_list):
    
            passage_pred_list = [pred_array[0,i] for i in range(pred_array.shape[1])]

            passage_labels = []
        
            for topic_index, pred_value in enumerate(passage_pred_list):
                
                topic_name = self.relevant_topics[topic_index]
                
                if pred_value != 0:

                    if self.topic_group == 'Super Topics':

                        passage_labels.append(topic_name)

                    else:

                        pred_super_topics = data_set['Pred Super Topics'][passage_index]

                        topic_group_name = self.topic_group.split()[-1]

                        # filter out topics whose families were not predicted
                        if topic_group_name in pred_super_topics:

                            passage_labels.append(topic_name)

            pred_labels_list.append(passage_labels)

        return pred_labels_list


    def predict(self, data_set, x_input):

        pred_arrays = list(self.classifier.predict(x_input))

        pred_lists = self.get_pred_labels_list(data_set, pred_arrays, self.topic_group) 

        data_set[f'Pred {self.topic_group}'] = pred_lists

        data_set[f'True {self.topic_group}'] = data_set[f'True {self.topic_group}'].str.split()


    def fit_and_pred(self):

        self.relevant_topics = self.topic_lists[self.topic_group]

        self.train_classifier()

        self.predict(data_set = self.train_set, x_input = self.x_train)
        self.predict(data_set = self.test_set, x_input = self.x_test)


    def one_hot_encode(self):

        for col in self.df.columns:

            if 'Super' in col or 'Children' in col:

                # add categorical columns
                self.df = pd.concat([self.df, self.df[col].str.get_dummies(sep=' ')], axis=1)

        # remove duplicated columns, for children with multiple parents 
        self.df = self.df.loc[:,~self.df.columns.duplicated()]


    def get_pred_vs_true(self, true_labels_df, pred_list):
        
        wanted_cols = [col for col in true_labels_df.columns if 'topic' in col.lower()]

        true_vs_pred_labels_df = true_labels_df[['passage_words'] + wanted_cols]
        
        true_vs_pred_labels_df[f'Pred {self.classification_stage}'] = pred_list

        return true_vs_pred_labels_df


class ConfusionMatrix:

    def __init__(self, topics, should_print = False):

        self.topics = topics
        self.should_print = should_print

    def get_cm_values(self, pred_vs_true):

        true_label_set_list = pred_vs_true.true_topics.tolist()
        pred_label_set_list = pred_vs_true.pred_topics.tolist()

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
        
        y_actu = pd.Categorical(y_true, categories=self.topics)
        y_pred = pd.Categorical(y_pred, categories=self.topics)

        cm = pd.crosstab(y_actu, y_pred, rownames=['True'], colnames = ['Prediction'], dropna=False) 

        return cm


    def check_worst(self,cm, labels_df):
        """
        Given confusion matrix, calculate which topic statistically performs the worst, and produce those examples.
        """

        cm_norm = cm.div(cm.sum(axis=1), axis=0)

        if self.should_print:
        
            print(cm_norm.round(2))

        TP_rates = {}
        for topic in self.topics:
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

    def __init__(self, topic_names, topic_counts, row_lim, expt_num, 
                none_ratio, use_expanded_topics = False, should_print = False,
                chosen_topics = None):
    
        self.row_lim = row_lim
        self.expt_num = expt_num
        self.topic_names = topic_names
        self.none_ratio = none_ratio
        self.should_print = should_print
        self.topic_counts = topic_counts
        self.chosen_topics = chosen_topics
        self.use_expanded_topics = use_expanded_topics


    def get_scores(self, cm):

        meaningful_topics = list(cm.columns)

        meaningful_topics.remove("None")

        precision_dict, recall_dict, f1score_dict = {}, {}, {}

        for topic in meaningful_topics:

            TP = cm.loc[topic,topic]
            
            FN = cm.loc[topic].sum() - TP

            recall = TP/(TP + FN)

            FP = cm[topic].sum() - TP - cm.loc["None",topic]

            precision = TP/(TP + FP)
            
            f1score = 2 * (precision * recall)/(precision + recall)
            
            precision_dict[topic], recall_dict[topic], f1score_dict[topic] = precision, recall, f1score

        scores = {
            "recall":recall_dict, 
            'f1score':f1score_dict,
            'precision':precision_dict, 
        } 

        return scores


    def get_stats_df(self, cm, dataset = "None"):

        scores = self.get_scores(cm)

        row_lim = self.row_lim
        expt_num = self.expt_num
        none_ratio = self.none_ratio

        recall_dict = scores['recall']
        f1score_dict = scores['f1score']
        precision_dict = scores['precision']

        topic_stats_df = pd.DataFrame(self.topic_counts,columns=['Topic','Occurrences'])
        
        total_occurrences = sum(occurrences for topic, occurrences in self.topic_counts)

        assert total_occurrences == topic_stats_df.Occurrences.sum()

        topic_stats_df['Proportion'] = topic_stats_df.Occurrences/total_occurrences

        topic_stats_df['Precision'] = topic_stats_df['Topic'].map(precision_dict)
        # topic_stats_df['Precision_using_series'] = pd.Series(precision_dict) # this way the dict keys need to be index of df
        topic_stats_df['Recall'] = topic_stats_df['Topic'].map(recall_dict)
        topic_stats_df['F1score'] = topic_stats_df['Topic'].map(f1score_dict)

        over_all_stats = {}

        over_all_stats['Recall'] = (topic_stats_df.Recall * topic_stats_df.Proportion).sum()
        over_all_stats['F1score'] = (topic_stats_df.F1score * topic_stats_df.Proportion).sum()
        over_all_stats['Precision']= (topic_stats_df.Precision * topic_stats_df.Proportion).sum()
        
        over_all_stats['Topic']= 'Overall'
        over_all_stats['Proportion'] = topic_stats_df.Proportion.sum() # ibid
        over_all_stats['Occurrences'] = topic_stats_df.Occurrences.sum() # exlcuding none occurrences

        topic_stats_df = topic_stats_df.append(over_all_stats, ignore_index=True)
        
        # my_topics = ["Overall",'laws-of-judges-and-courts', 'prayer', 'procedures-for-judges-and-conduct-towards-them']
        my_topics = self.chosen_topics + ['Overall']
        
        selected_topics_df = topic_stats_df.loc[topic_stats_df['Topic'].isin(my_topics)]

        selected_topics_list = selected_topics_df['Topic'].to_list()

        selected_scores_list = selected_topics_df['F1score'].to_list()

        if self.should_print:
            # print(f'\n\n{dataset}\n')
            print(topic_stats_df.round(2))
            # print(f'\ntopics = ', selected_topics_list)
            # print(f'{dataset}_expt_{expt_num} =', selected_scores_list)

        return topic_stats_df


class Trainer:
    
    def __init__(self, classifier):
    
        self.classifier = classifier


    def train(self, x_train, y_train):
        
        classifier = self.classifier
        
        try:
            classifier.fit(x_train, y_train)

        except:        
            pass
            y_train = y_train.values.toarray()
            
            classifier.fit(x_train, y_train)

        if isinstance(classifier, sklearn.model_selection._search.GridSearchCV):
            print ("Best params:",classifier.best_params_)
            print("Best score:",classifier.best_score_)
        
        return classifier


class MultiStageClassifier:

    def __init__(self, expt_num, super_topics):

        self.expt_num = expt_num
        self.super_topics = super_topics

    def super_classify():
            
            # df = all data with topics and exp topics

            # super_topics = [top 4 or 5]

        # limited_df = without extraneous super topics

        # stage 1 result
        # pred_super_df  = df with pred super_topics

        # limiter = Limiter()

        # now stage 2
        return None



    def sub_classify():
        for super_topic in super_topics:
            pass

        result = None

        return result


class Evaluator:

    def __init__(self, results):

        self.results = results



class TopicCounter:


    def __init__(self):
    
        pass


    def get_counts(self, series, max_topics):

        topic_set_lst = series.to_list()

        all_topics_str = ' '.join(topic_set_lst)

        all_topics_lst = all_topics_str.split()

        counter = Counter(all_topics_lst)

        topic_counts = counter.most_common(max_topics) # must change from 4 **********************************

        return topic_counts


    def topic_limiter(self, row, permitted_topics):

        # this cell contains more topics than we might want
        old_passage_topics_string = row

        # compile list of this passage's topics, only including those which were top ranked
        new_passage_topics_list = [topic for topic in old_passage_topics_string.split() if topic in permitted_topics]
        
        # reconnect the topics in the list to form a string separated by spaces
        new_passage_topics_string = ' '.join(new_passage_topics_list)
        
        return new_passage_topics_string    