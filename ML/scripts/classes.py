import re
import nltk
import sklearn
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.model_selection
from unidecode import unidecode
import matplotlib.pyplot as plt

from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.stem import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# my_example_topics = ['prayer', 'procedures-for-judges-and-conduct-towards-them', 'learning', 'kings', 'hilchot-chol-hamoed', 'laws-of-judges-and-courts', 'laws-of-animal-sacrifices', 'financial-ramifications-of-marriage', 'idolatry', 'laws-of-transferring-between-domains']
# my_example_topics = ['prayer', 'procedures-for-judges-and-conduct-towards-them']
# my_example_topics = ['prayer']

stemmer = SnowballStemmer('english')

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
    def __init__(self, raw, num_topics, my_topics, should_clean = True, should_stem = False, should_remove_stopwords = True):
        self.raw = raw
        self.my_topics = my_topics
        self.num_topics = num_topics
        self.should_stem = should_stem
        self.should_clean = should_clean
        self.should_remove_stopwords = should_remove_stopwords

    def get_my_topics(self,all_topics):
        all_topics_list = all_topics.split()
        sublist = [topic for topic in all_topics_list if topic in self.my_topics]
        result = ' '.join(sublist)
        return result

    def preprocess_dataframe(self):
        df = self.raw
        print('Original shape:',df.shape)
        df = df.drop_duplicates()
        print('Without duplicates:',df.shape)
        df = df.dropna()
        print('Without nulls:',df.shape)
        df = df.set_index('Ref',drop=True)
        df = df[['En','Topics']]
        df = df.rename(columns={'En': 'passage_text'})
        df['Topics'] = df['Topics'].apply(self.get_my_topics)
        df['Topics'].replace('', np.nan, inplace=True)
        df = df.dropna()
        df = pd.concat([df, df.pop('Topics').str.get_dummies(sep=' ')], axis=1)
        # df_wanted_rows = data_raw[~(df_all_rows[my_example_topics] == 0).all(axis=1)]
        return df

    def categories(self):
        return list(self.preprocess_dataframe().columns.values)[1:]

    def get_topic_counts(self):
        df = self.preprocess_dataframe()
        # categories = list(df.columns.values)
        # categories = categories[1:]
        counts = []
        for category in self.categories:
            counts.append((category, df[category].sum()))
        df_stats = pd.DataFrame(counts, columns=['category', 'number of passages'])
        return df_stats

    # def show_simple_plot(self):
        # figure = plt.figure()
        # figure = plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        # return figure


    def show_topic_counts(self):
        df = self.preprocess_dataframe()
        categories = list(df.columns.values)
        categories = categories[1:]
        sns.set(font_scale = 2)
        figure = plt.figure(figsize=(15,8))

        ax= sns.barplot(categories, df.iloc[:,1:].sum().values)

        plt.title("Passages in each category", fontsize=24)
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

    def cleanHtml(self,sentence):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', str(sentence))
        return cleantext


    def cleanPunc(self,sentence): #function to clean the word of any punctuation or special characters
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n"," ")
        return cleaned


    def keepAlpha(self,sentence):
        sentence = unidecode(sentence)
        alpha_sent = ""
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent

    def clean_text(self):
        data = self.preprocess_dataframe()
        data['passage_text'] = data['passage_text'].str.lower()
        data['passage_text'] = data['passage_text'].apply(self.cleanHtml)
        data['passage_text'] = data['passage_text'].apply(self.cleanPunc)
        data['passage_text'] = data['passage_text'].apply(self.keepAlpha)
        return data
    
    def stopword_cleaner(self,sentence):
        stop_words = set(stopwords.words('english'))
        re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
        sentence = re_stop_words.sub(" ", sentence)
        return sentence

    def remove_stopwords(self):
        data = self.clean_text()
        if self.should_remove_stopwords:
            data['passage_text'] = data['passage_text'].apply(self.stopword_cleaner)
        return data
    
    def stemmer(self,sentence):
        stemmer = SnowballStemmer("english")
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence


    def stem_words(self):
        data = self.clean_text()
        if self.should_stem:
            data['passage_text'] = data['passage_text'].apply(self.stemmer)
        return data

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


    def _stem_text(self, sentence):
        # instantiate stemmer class
        # stemmer = SnowballStemmer('english')

        # stem sentence
        sentence = ' '.join(stemmer.stem(word) for word in sentence.split())
        return sentence


    def _clean_columns(self):
        
        df = self._add_ref_features()
        
        if self.should_clean:
            df.En = df.En.apply(self._clean_text)
        
        if self.should_stem:
            df.En = df.En.apply(self._stem_text)
        
        return df

    def _add_topic_columns(self):
        df = self._clean_columns()
        start_time = datetime.now()
        df = pd.concat([df, df['Topics'].str.get_dummies(sep=' ')], axis=1)
        cols = ['Ref', 
        # 'ref_features',
        'En','Topics'] + my_example_topics
        df = df[cols]
        # df = df.loc[df['prayer'] + df['procedures-for-judges-and-conduct-towards-them'] > 0]
        # df = df.loc[df['prayer'] > 0]
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
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag',max_iter=1000), n_jobs=1)),
                ])
        }

    def get_pipeline(self):
        return self.pipelines[self.model_code]


# # class Classifier
# # class Evaluator
