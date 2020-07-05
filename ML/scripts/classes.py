import re
import nltk
import sklearn
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from string import printable
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

nltk.download('stopwords')
stemmer = SnowballStemmer('english')

class DataManager:
    """
    1 input:
    - raw data: pandas dataframe (heneforth "df")

    tasks:
    
    5 critical:
    X remove nulls and duplicates
    X get top topics 
    - in topics col, keep only wanted topics
    - clean text 
    - one-hot-encode the list of wanted topics
    
    3 less important
    - add parsed_Ref column to show just relevant subject
        - e.g. "Mishna Torah, Shabbat, 4:7" --> "shabbat"
    - divide labeled from unlabeled.
    - within labeled, split into train and test set.
    """
    def __init__(self, raw_df, num_topics, should_clean = True, should_stem = False, should_remove_stopwords = True):
        self.raw_df = raw_df
        self.num_topics = num_topics
        self.should_stem = should_stem
        self.should_clean = should_clean
        self.should_remove_stopwords = should_remove_stopwords


    def remove_junk_rows(self):
        df = self.raw_df
        # how many rows and columns
        print('Original shape:',df.shape)
        # remove repeats
        df = df.drop_duplicates()
        print('Without duplicates:',df.shape)
        # remove empty cells
        df = df.dropna()
        print('Without nulls:',df.shape)
        return df


    def _get_top_topics(self):
        if getattr(self, "top_topics", None):
            return self.top_topics
        else:
            # count all topics that appear without having removed junk rows
            df = self.raw_df
            # make str out of all topic lists in topics column
            all_topics_list = ' '.join(df['Topics'].tolist()).split()
            # init dict
            topic_counts = {}
            # loop thru all topic occurrences
            for topic in all_topics_list:
                # increment if seen already
                if topic in topic_counts:
                    topic_counts[topic] += 1
                # init if not seen yet
                else:
                    topic_counts[topic] = 1
            # rank the entries by most frequently occurring first
            top_topic_counts = {k: v for k, v in sorted(topic_counts.items(), key=lambda item: item[1],reverse=True)}
            # convert dict {'prayer':334, etc} to list [('prayer',334), etc]
            topic_tuples = list(top_topic_counts.items())
            # select only the highest ranking
            top_topic_tuples = topic_tuples[:self.num_topics]
            # extract only the names of these topics, whilst dropping the number of occrurences 
            top_topics_list = [topic_tuple[0] for topic_tuple in top_topic_tuples]
            self.top_topics = top_topics_list
        return top_topics_list


    def topic_selector(self,row):
        # this cell contains more topics than we might want
        all_topics_list = row
        # call attribute which stored top topics
        # top_topics_list = self.top_topics()
        top_topics_list = self._get_top_topics()
        # keep only the topics which are popular
        reduced_topics_list = [topic for topic in all_topics_list.split() if topic in top_topics_list]
        # reconnect the topics in the list to form a string separated by spaces
        reduced_topics_string = ' '.join(reduced_topics_list)
        return reduced_topics_string


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


    # def clean_text(self):
    #     data = self.preprocess_dataframe()
    #     data['passage_text'] = data['passage_text'].str.lower()
    #     data['passage_text'] = data['passage_text'].apply(self.cleanHtml)
    #     data['passage_text'] = data['passage_text'].apply(self.cleanPunc)
    #     data['passage_text'] = data['passage_text'].apply(self.keepAlpha)
    #     return data
    

    def stopword_cleaner(self,sentence):
        stop_words = set(stopwords.words('english'))
        re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
        sentence = re_stop_words.sub(" ", sentence)
        return sentence


    # def remove_stopwords(self):
    #     data = self.clean_text()
    #     if self.should_remove_stopwords:
    #         data['passage_text'] = data['passage_text'].apply(self.stopword_cleaner)
    #     return data

    
    def stemmer(self,sentence):
        stemmer = SnowballStemmer("english")
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence


    # def stem_words(self):
    #     data = self.remove_stopwords()
    #     if self.should_stem:
    #         data['passage_text'] = data['passage_text'].apply(self.stemmer)
    #     return data

    def preprocess_dataframe(self):
        df = self.remove_junk_rows()
        # use Ref as index instead of number
        df = df.set_index('Ref',drop=True)
        # keep only these columns
        df = df[['En','Topics']]
        # add more descriptive name
        df = df.rename(columns={'En': 'passage_text'})
        # keep only topics that i want to study
        df['true_topics'] = df.pop('Topics').apply(self.topic_selector)
        # remove rows which don't have my topics
        df['true_topics'].replace('', np.nan, inplace=True)
        # remove casualties
        df = df.dropna()
        # one hot encode each topic
        df = pd.concat([df, df['true_topics'].str.get_dummies(sep=' ')], axis=1)
        # make topic string into list
        df['true_topics'] = df['true_topics'].str.split()
        # clean passage text
        df['passage_text'] = df['passage_text'].str.lower()
        df['passage_text'] = df['passage_text'].apply(self.cleanHtml)
        df['passage_text'] = df['passage_text'].apply(self.cleanPunc)
        df['passage_text'] = df['passage_text'].apply(self.keepAlpha)
        # remove stopwords, if you so chose  
        if self.should_remove_stopwords:
            df['passage_text'] = df['passage_text'].apply(self.stopword_cleaner)
        # stem words, if you so chose  
        if self.should_stem:
            df['passage_text'] = df['passage_text'].apply(self.stemmer)
        return df


    # def one_hot_encode(self):
    #     df = self.preprocess_dataframe()
    #     # df_wanted_rows = data_raw[~(df_all_rows[my_example_topics] == 0).all(axis=1)]
    #     return df


    def get_topic_counts(self):
        df = self.preprocess_dataframe()
        # categories = list(df.columns.values)
        # categories = categories[1:]
        counts = []
        
        for category in self.my_topics:
        # for category in self.categories:
            counts.append((category, df[category].sum()))
        df_stats = pd.DataFrame(counts, columns=['category', 'number of passages'])
        return df_stats


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



    def _get_labeled(self):
        df = self._add_topic_columns()
        print('Shape of labeled data:',df.shape)
        return df[df.Topics.notnull()]
        

    def _get_unlabeled(self):
        df = self._add_topic_columns()
        print('Shape of unlabeled data:',df.shape)
        return df[df.Topics.isnull()]


    # def get_train_and_test(self):
    #     labeled_data = self._get_labeled()
    #     train, test = labeled_data[:-1], labeled_data[-5:]
    #     train, test = sklearn.model_selection.train_test_split(labeled_data,random_state=42, test_size=0.33, 
    #     shuffle=True
    #     )
    #     return train, test


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
