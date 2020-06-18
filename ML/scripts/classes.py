#%%
import matplotlib.pyplot as plt
import pandas as pd 
from gensim.parsing.preprocessing import STOPWORDS
from bs4 import BeautifulSoup
import sklearn
import re

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

    5 outputs: 
        - train passages (df) 
        - train topics (df) 
        - test passages (df)
        - test topics (df)
        - unlabeled passages (df)
    """
    def __init__(self, raw):
        self.raw = raw

    def _select_columns(self):
        df = self.raw
        return df[['Ref','En','Topics']]

    def _remove_null(self):
        df = self._select_columns()
        rows_before = df.shape[0]
        df = df.dropna(subset=['Ref', 'En'])
        rows_after = df.shape[0]
        print(f"Dropped {rows_before - rows_after} nulls!")
        return df

    def _remove_duplicates(self):
        df = self._remove_null()
        rows_before = df.shape[0]
        df = df.drop_duplicates()
        rows_after = df.shape[0]
        print(f"Dropped {rows_before - rows_after} duplicates!")
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
        return df.join(pd.get_dummies(df['Topics']))

    def _topic_stats(self):
        df = self._add_topic_columns()
        df_topics = df.drop(['Ref', 'ref_features','En','Topics'], axis=1)
        counts = []
        topics = list(df_topics.columns.values)
        for topic in topics:
            counts.append((topic, df_topics[topic].sum()))
        df_stats = pd.DataFrame(counts, columns=['topic', 'occurrences'])
        df_stats = df_stats.sort_values(by=['occurrences'], ascending=False)#[:10,:]
        return df_stats[:6]
        # df_stats.plot(x='topic', y='occurrences', kind='bar', legend=False, grid=True, figsize=(8, 5))
        # plt.title("Number of comments per category")
        # plt.ylabel('# of Occurrences', fontsize=12)
        # plt.xlabel('category', fontsize=12)
        # plt.show

    def _get_labeled(self):
        df = self._add_topic_columns()
        print('labeled')
        return df[df.Topics.notnull()]
        
    def _get_unlabeled(self):
        df = self._add_topic_columns()
        print('unlabeled')
        return df[df.Topics.isnull()]

    def train_test_split(self,labeled):
        labeled_data = self._labeled_unlabeled_split()['labeled']
        return sklearn.model_selection.train_test_split(labeled_data)


class Trainer:
    """
    2 inputs:
        - training set
        - model untrained

    what this class does:
        - trains model to fit training data

    1 output:
        - trained model
    """
    def __init__(self, train_df, model):
        self.train_df = train_df
        self.model = model

    def _select_columns(self):
        df = self.raw
        return df[['Ref','En','Topics']]


# class Classifier
# class Evaluator
