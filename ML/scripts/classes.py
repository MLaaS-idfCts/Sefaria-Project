# %matplotlib inline
import re
import sklearn
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.model_selection
import matplotlib.pyplot as plt

from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# my_example_topics = ['prayer', 'procedures-for-judges-and-conduct-towards-them', 'learning', 'kings', 'hilchot-chol-hamoed', 'laws-of-judges-and-courts', 'laws-of-animal-sacrifices', 'financial-ramifications-of-marriage', 'idolatry', 'laws-of-transferring-between-domains']
my_example_topics = ['prayer', 'procedures-for-judges-and-conduct-towards-them']

class DataManager:
    """
    1 input:

    - raw data: pandas dataframe (heneforth "df")

    3 tasks:

    - clean data
        - keep only three columns: Ref, En, and Topics
        - remove rows with null Ref or En
        - remove duplicated rows
        - add parsed_Ref column to show just relevant subject
            - e.g. "Mishna Torah, Shabbat, 4:7" --> "shabbat"
        - clean En column

    - breakdown topics
        - one-hot-encode the list from Topics column
        - present number of topic occurrences

    -divide data
        - divide labeled from unlabeled.
        - within labeled, split into train and test set.

    """
    def __init__(self, raw, num_topics):
        self.raw = raw
        self.num_topics = num_topics

    def _select_columns(self):
        df = self.raw
        return df[[
            'Ref',
            'En','Topics']]

    def _remove_null(self):
        df = self._select_columns()
        rows_before = df.shape[0]
        df = df.dropna(subset=['Ref', 'En'])
        rows_after = df.shape[0]
        # print(f"Dropped {rows_before - rows_after} nulls!")
        return df

    def _remove_duplicates(self):
        df = self._remove_null()
        rows_before = df.shape[0]
        df = df.drop_duplicates()
        rows_after = df.shape[0]
        # print(f"Dropped {rows_before - rows_after} duplicates!")
        return df

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
        df = self._remove_duplicates()
        df['ref_features'] = df.Ref.apply(self._get_ref_features)
        return df

    def _clean_text(self, sentence):

        # HTML decoding
        sentence = BeautifulSoup(sentence, "lxml").text 
        
        # lowercase text
        sentence = sentence.lower() 

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        # Removing stopwords
        sentence = ' '.join(word for word in sentence.split() if word not in STOPWORDS) # delete stopwors from text

        return sentence

    def _clean_columns(self):
        df = self._add_ref_features()
        df.En = df.En.apply(self._clean_text)
        # df.En = self._clean_text(df.En)
        return df

    def _add_topic_columns(self):
        df = self._clean_columns()
        start_time = datetime.now()
        df = pd.concat([df, df['Topics'].str.get_dummies(sep=' ')], axis=1)
        cols = ['Ref', 
        # 'ref_features',
        'En','Topics'] + my_example_topics
        df = df[cols]
        df = df.loc[df['prayer'] + df['procedures-for-judges-and-conduct-towards-them'] > 0]
        return df


    def get_top_topics(self):

        df = self._add_topic_columns()
        df_topics = df.drop(['Ref', 'ref_features','En','Topics',
        # 'Extended-topics'
        ], axis=1)

        counts = []
        topics = list(df_topics.columns.values)

        print("\nCounting occurrences of each topic")
        for topic in tqdm(topics):
            counts.append((topic, df_topics[topic].sum()))

        df_stats = pd.DataFrame(counts, columns=['topic', 'occurrences'])
        df_stats_sorted = df_stats.sort_values(by=['occurrences'], ascending=False)
        top_topics_df = df_stats_sorted[:self.num_topics]
        return top_topics_df


    def _get_labeled(self):
        df = self._add_topic_columns()
        print('Shape of labeled data:',df.shape)
        return df[df.Topics.notnull()]
        
    def _get_unlabeled(self):
        df = self._add_topic_columns()
        print('Shape of unlabeled data:',df.shape)
        return df[df.Topics.isnull()]

    def get_train_and_test(self):
        labeled_data = self._get_labeled()
        train, test = labeled_data[:-1], labeled_data[-5:]
        train, test = sklearn.model_selection.train_test_split(labeled_data,random_state=42, test_size=0.33, 
        shuffle=True
        )
        return train, test


class PipelineFactory:


    def __init__(self, model_code):
        self.model_code = model_code
        self.stop_words = set(stopwords.words('english'))
        self.pipelines = {
            "MultNB":Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=self.stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
                ]),
            "LinSVC":Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=self.stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
                ]),
            "LogReg":Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=self.stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
                ])
        }

    def get_pipeline(self):
        return self.pipelines[self.model_code]


# # class Classifier
# # class Evaluator
