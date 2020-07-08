import re
import nltk
import scipy
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection

from bs4 import BeautifulSoup
from tqdm import tqdm
from string import printable
from datetime import datetime
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
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
    def __init__(
        self, raw_df, num_topics, 
        should_clean = True, should_stem = False, should_remove_stopwords = True, count_none = False):
        
        self.raw_df = raw_df
        self.num_topics = num_topics
        self.should_stem = should_stem
        self.should_clean = should_clean
        self.count_none = count_none
        self.should_remove_stopwords = should_remove_stopwords


    def remove_junk_rows(self):
        
        if isinstance(getattr(self, "without_junk_rows", None), pd.DataFrame):
            return self.without_junk_rows
        else:
            df = self.raw_df
            # remove repeats
            df = df.drop_duplicates()
            # remove empty cells
            df = df.dropna()
            # store as attribute
            self.without_junk_rows = df
        return df


    def _get_top_topics(self):

        if getattr(self, "top_topics", None):
            return self.top_topics
        
        else:
            # count all topics that appear *after* having removed junk rows
            df = self.remove_junk_rows()

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
            top_topics_list = [topic_tuple[0] for topic_tuple in top_topic_tuples] + ['None']
            
            # order by topic name
            top_topics_list.sort()
            
            # store as attribute
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
        
            # keep only these columns
            df = df[['En','Topics']]
        
            # add more descriptive name
            df = df.rename(columns={'En': 'passage_text'})
        
            # keep only topics that i want to study
            df['true_topics'] = df.pop('Topics').apply(self.topic_selector)
        
            # remove rows which don't have my topics
            df['true_topics'].replace('', 'None', inplace=True)
            # df['true_topics'].replace('', np.nan, inplace=True)
        
            # remove casualties
            df = df.dropna()
        
            # one hot encode each topic
            df = pd.concat([df, df['true_topics'].str.get_dummies(sep=' ')], axis=1)

            # order columns, starting with most frequently occurring topic
            ordered_columns = ['passage_text', 'true_topics'] + self.top_topics
            df = df[ordered_columns]
        
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

            self.preprocessed_dataframe = df

        return df


    def get_topic_counts(self):
        if getattr(self, "topic_counts", None):
            return self.topic_counts
        else:
            
            df = self.preprocess_dataframe()
            counts = []
            for topic in self.top_topics:
                counts.append((topic, df[topic].sum()))
            topic_counts = pd.DataFrame(counts, columns=['Topic', 'Occurrences'])
            topic_counts = topic_counts.sort_values(by='Occurrences', ascending=False, na_position='last')
            topic_counts = topic_counts.reset_index(drop=True)
            
            total_occurrences = topic_counts.Occurrences.sum()
            topic_counts['Proportion'] = round(topic_counts.Occurrences/total_occurrences,2)

            topic_counts = topic_counts.sort_values(by='Topic')

        return topic_counts


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

class ConfusionMatrix:
    def __init__(self, result, top_topics):
        self.result = result
        self.top_topics = top_topics

    def get_values(self):

        # return 12

    # if True:
        result = self.result
        top_topics = self.top_topics

        true_label_lists = result.true_topics.tolist()
        pred_label_lists = result.pred_topics.tolist()

        assert len(true_label_lists) == len(pred_label_lists)

        num_passages = len(true_label_lists)

        y_true = []
        y_pred = []

        for i in range(num_passages):

            true_label_list = []
            pred_label_list = []
            
            try:
                true_label_list = true_label_lists[i]
            except:
                pass
            
            try:
                pred_label_list = pred_label_lists[i]
            except:
                pass

            # 0) NULL CASE --> No true or pred labels 
            if len(pred_label_list) == 0 and len(pred_label_list) == 0:
                    y_true.append('None')
                    y_pred.append('None')
        
            # 1) MATCH --> true label == pred label 
            for true_label in true_label_list:
                if true_label in pred_label_list:
                    y_true.append(true_label)
                    y_pred.append(true_label)

            # 2) FALSE NEGATIVE --> true label was not predicted
                else:
                    y_true.append(true_label)
                    y_pred.append("None")

            # 3) FALSE POSITIVE --> pred label was not true
            for pred_label in pred_label_list:
                if pred_label not in true_label_list:
                    y_true.append("None")
                    y_pred.append(pred_label)
        
        # topics_list = ['None'] + top_topics
        topics_list = top_topics
                
        y_actu = pd.Categorical(y_true, categories=topics_list)
        y_pred = pd.Categorical(y_pred, categories=topics_list)

        cm = pd.crosstab(y_actu, y_pred, rownames=['True'], colnames = ['Pred'], dropna=False) 
        # print(cm)
        cm = cm.values
        return cm

class Predictor:
    def __init__(self, classifier, test, x_test, top_topics):
        self.classifier = classifier
        self.test = test
        self.x_test = x_test
        self.top_topics = top_topics
    
    def get_result(self):
        
        classifier = self.classifier
        top_topics = self.top_topics
        test = self.test
        x_test = self.x_test

        # make predictions
        test_predictions = classifier.predict(x_test)
        # train_predictions = classifier.predict(x_train)

        # convert csc matrix into list of csr matrices, e.g. preds_list = [[0,1,0,1,0],[0,0,0,1,0],...,[1,0,0,0,0]]
        # one matrix per passage, e.g. [0,0,0,1,1] inidicates this passage corrseponds to the last two topics
        preds_list = list(test_predictions)
        
        # init list of sublists, one sublist per passage, 
        # e.g. pred_labels_list = [['prayer'],['prayer','judges'],['moses']]
        pred_labels_list = []
        
        # loop thru each matrix in list, again one matrix per passage, 
        for array in preds_list:
        
            if isinstance(array, scipy.sparse.csr.csr_matrix) or isinstance(array, np.int64) or isinstance(array, scipy.sparse.lil.lil_matrix):
                # array = array.tolil().data.tolist()
                array = [array[0,i] for i in range(array.shape[1])]
        
            # init topics list for this row, e.g. passage_labels = ['prayer', 'moses']
            passage_labels = []
        
        
            # if 1 occurs in ith element in the array, record ith topic
            for topic_index, pred_value in enumerate(list(array)):
                if pred_value != 0:
                    passage_labels.append(top_topics[topic_index])
            pred_labels_list.append(passage_labels)

        test['pred_topics'] = pred_labels_list

        cols=['passage_text','true_topics','pred_topics']

        # topics_comparison = test[['true_topics','pred_topics']]
        pred_vs_true = test[cols]
        return pred_vs_true


class DataConverter:

    def __init__(self, data):
    
        self.data = data

    def get_datasets(self):

        data = self.data

        # randomly split into training and testing sets
        train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)

        # init column for topic predictions
        test['pred_topics'] = None

        # select just the words of each passage
        train_text = train['passage_text']
        test_text = test['passage_text']

        # init a vectorizer that will convert string of words into numerical format
        # vectorizer = TfidfVectorizer(max_features=5000,stop_words='english')
        vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,smooth_idf=1, sublinear_tf=1 )
        # vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')

        # create document-term matrix, i.e. numerical version of passage text
        # Note: We only fit with training data, but NOT with testing data, because testing data should be "UNSEEN"
        x_train = vectorizer.fit_transform(train_text)
        x_test = vectorizer.transform(test_text)

        # topics columns, with 0/1 indicating if the topic of this column relates to that row's passage
        y_train = train.drop(labels = ['passage_text','true_topics'], axis=1)
        y_test = test.drop(labels = ['passage_text','true_topics'], axis=1)

        return x_train, x_test, y_train, y_test, test


class Scorer:
    def __init__(self, cm, top_topics, topic_counts):
    
        self.cm = cm
        self.topic_counts = topic_counts
        self.top_topics = top_topics

    def get_result(self):

        cm = self.cm
        topic_counts = self.topic_counts
        top_topics = self.top_topics

        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        
        np.seterr(divide='ignore', invalid='ignore')

        # precision = how many predicted were correct
        precision = TP/(TP+FP)
        
        # recall = how many correct were predicted
        recall = TP/(TP+FN)
        
        # f1 score = harmonic mean of precision and recall
        f1score = TP/(TP+(FP+FN)/2)
        
        score_df = pd.DataFrame({
            'Topic': top_topics,
            'Count': topic_counts.Occurrences.tolist(), 
            'Proportion': topic_counts.Proportion.tolist(), 
            'Precision': precision, 
            'Recall': recall, 
            'F1score': f1score
            })
        # score_df = score_df.set_index(['Topic'], drop=True)
        score_df = score_df.set_index(['Topic'])
        score_df = score_df.loc[score_df.index.isin(top_topics)]
        # score_df = score_df.loc[score_df.index.isin(top_topics)]
        score_df = score_df.sort_values(by=['F1score', 'Precision', 'Recall'], ascending=False)
        score_df = score_df.reset_index()

        # overall scores 
        overall_precision = (score_df.Precision * score_df.Proportion).sum()
        overall_recall = (score_df.Recall * score_df.Proportion).sum()
        overall_f1score = (score_df.F1score * score_df.Proportion).sum()

        score_df.loc['OVERALL -->'] = ['',score_df.Count.sum(),score_df.Proportion.sum(),overall_precision,overall_recall,overall_f1score]#print(f"Overall: \nPrecision: {overall_precision}, \nRecall: {overall_recall}, \nF1score: {overall_f1score}")

        return score_df

# # class Classifier
# # class Evaluator
