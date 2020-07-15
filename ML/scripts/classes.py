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

class DataManager:

    def __init__(
        self, 
        raw_df, 
        num_topics, 
        none_ratio, 
        should_limit_nones = False,
        should_stem = False, 
        should_clean = True, 
        keep_all_nones = False,
        should_remove_stopwords = False, 
        ):
        
        self.raw_df = raw_df
        self.should_limit_nones = should_limit_nones
        self.none_ratio = none_ratio
        self.num_topics = num_topics
        self.should_stem = should_stem
        self.should_clean = should_clean
        self.keep_all_nones = keep_all_nones
        self.should_remove_stopwords = should_remove_stopwords

    
    def get_top_topic_counts(self, df):

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
        topic_counts_dict = {
                                k: v for k, v in sorted(topic_counts.items(), 
                                key=lambda item: item[1],
                                reverse=True)
                            }
        
        topic_counts_list = [(k, v) for k, v in topic_counts_dict.items()] 

        top_topic_counts_list = topic_counts_list[:self.num_topics]

        return top_topic_counts_list


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


    def reduce_topics(self, df):

        # keep only topics that i want to study
        df['true_topics'] = df.pop('Topics').apply(self.topic_selector)

        # **************************************************
        # df['true_topics'].replace('', 'None', inplace=True)
        # **************************************************

        return df


    def tidy_up(self, df):

        # clean passage text
        if self.should_clean:
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


    def get_reduced_topics_df(self):

        if getattr(self, "reduced_topics_df", None):

            return self.reduced_topics_df
        
        else:
            
            all_topics_df = self.preprocess_dataframe()

            self.top_topic_counts_list = self.get_top_topic_counts(all_topics_df)

            self.top_topics_list = [topic_tuple[0] for topic_tuple in self.top_topic_counts_list]

            # eliminate topic rows with no top topics
            reduced_topics_df = self.reduce_topics(all_topics_df)

            # store as attribute
            self.reduced_topics_df = reduced_topics_df

        return reduced_topics_df


    def limit_nones(self, df):

        # place nones last
        df = df.sort_values(by='true_topics', ascending=False)

        # calc how many nones there are
        num_nones = df.loc[df['true_topics'] == ""].shape[0]

        # most commonly occurring topic
        top_topic_name = self.top_topic_counts_list[0][0]

        # num of occurrences of most popular topic 
        top_topic_counts = self.top_topic_counts_list[0][1]

        # init
        nones_to_drop = 0
        nones_to_keep = num_nones
        
        if self.none_ratio == 'all':

            pass
            
        else:

            # compute num of nones to keep based upon ratio
            nones_to_keep = int(top_topic_counts * self.none_ratio)


            # check there are more nones than the computed limit
            if nones_to_keep <= num_nones:
                    
                # calc how many nones to drop
                nones_to_drop = num_nones - nones_to_keep

            # remove final excess 'none' rows
            df = df.iloc[:-1 * nones_to_drop]

        # update list of topic counts
        top_topic_counts_list = self.top_topic_counts_list + [('None',nones_to_keep)]
        
        # ensure 'None' is placed at the beginning iff it has the most
        top_topic_counts_list.sort(key=lambda x:x[1],reverse=True)

        self.top_topic_counts_list = top_topic_counts_list

        self.top_topics_list = [topic_tuple[0] for topic_tuple in self.top_topic_counts_list]
        
        return df

    
    def one_hot_encode(self, df):

        # one hot encode each topic
        df = pd.concat([df, df['true_topics'].str.get_dummies(sep=' ')], axis=1)

        # # don't need a col for "None" because it is just the absence of all substantice topics
        # del df['None']

        # make topic string into list
        df['true_topics'] = df['true_topics'].str.split()

        return df


    def topic_selector(self, row):

        # this cell contains more topics than we might want
        old_passage_topics_string = row

        # keep only the topics which are popular
        passage_topics_list = [topic for topic in old_passage_topics_string.split() if topic in self.top_topics_list]
        
        # reconnect the topics in the list to form a string separated by spaces
        new_passage_topics_string = ' '.join(passage_topics_list)
        
        return new_passage_topics_string


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
        
            # make more descriptive name
            df = df.rename(columns={'En': 'passage_text'})
    
            self.preprocessed_dataframe = df

        return df


    # def get_topic_counts(self):
    #     if getattr(self, "topic_counts_list", None):
    #         return self.topic_counts_list
    #     else:
            
    #         df = self.preprocess_dataframe()
    #         counts = []
    #         for topic in self.top_topics:
    #             counts.append((topic, df[topic].sum()))
    #         topic_counts = pd.DataFrame(counts, columns=['Topic', 'Occurrences'])

    #     return topic_counts


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
    def __init__(self, 
    # result, 
    top_topics, should_print = False):
        # self.result = result
        self.top_topics = top_topics
        self.should_print = should_print

    def get_cm_values(self, pred_vs_true):

        top_topics = self.top_topics

        true_label_lists = pred_vs_true.true_topics.tolist()
        pred_label_lists = pred_vs_true.pred_topics.tolist()

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
        if self.should_print:
            print(cm)
        return cm



class Predictor:
    def __init__(self, classifier, 
    # train, test, 
    top_topics):
    # def __init__(self, classifier, test, x_test, top_topics):
        
        self.classifier = classifier
        # self.test = test
        # self.train = train
        # self.x_test = x_test
        self.top_topics = top_topics
    

    def get_preds_list(self, x_input):

        classifier = self.classifier
        top_topics = self.top_topics

        # make predictions
        predictions = classifier.predict(x_input)

        # convert csc matrix into list of csr matrices, e.g. preds_list = [[0,1,0,1,0],[0,0,0,1,0],...,[1,0,0,0,0]]
        # one matrix per passage, e.g. [0,0,0,1,1] inidicates this passage corrseponds to the last two topics
        preds_list = list(predictions)
        
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

        return pred_labels_list


    def get_pred_vs_true(self, passage_labels_df, pred_list):
        
        # test = self.test
        pred_labels_list = pred_list

        passage_labels_df['pred_topics'] = pred_labels_list

        cols=['passage_text','true_topics','pred_topics']

        pred_vs_true = passage_labels_df[cols]
        
        # print(pred_vs_true.sample(30).to_string(index=False))
        # print(pred_vs_true.sample(30))

        return pred_vs_true



class DataSplitter:

    def __init__(self, data_df):
    
        self.data_df = data_df

    def get_datasets(self, vectorizer):

        # arrange in ordr of index for passage
        self.data_df.sort_values(by='Ref')
        # randomly split into training and testing sets
        train, test = train_test_split(
            self.data_df, 
            shuffle=True,
            test_size=0.30, 
            random_state=42, 
        )
        #print("Time taken for train test split:\n", datetime.now() - start_time)
        start_time = datetime.now()

        # select just the words of each passage
        train_text = train['passage_text']
        test_text = test['passage_text']


        # create document-term matrix, i.e. numerical version of passage text
        # Note: We only fit with training data, but NOT with testing data, because testing data should be "UNSEEN"
        x_train = vectorizer.fit_transform(train_text)
        x_test = vectorizer.transform(test_text)
        #print("Time taken for vectorizer:\n", datetime.now() - start_time)
        start_time = datetime.now()

        # topics columns, with 0/1 indicating if the topic of this column relates to that row's passage
        y_train = train.drop(labels = ['passage_text','true_topics'], axis=1)
        y_test = test.drop(labels = ['passage_text','true_topics'], axis=1)

        return train, test, x_train, x_test, y_train, y_test



class Scorer:

    def __init__(self, top_topics, topic_counts, row_lim, expt_num, none_ratio, should_print = False):
    
        self.row_lim = row_lim
        self.expt_num = expt_num
        self.top_topics = top_topics
        self.none_ratio = none_ratio
        self.should_print = should_print
        self.topic_counts = topic_counts


    def get_scores(self, cm):

        topic_counts = self.topic_counts

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

        scores = {
            'precision':precision, 
            "recall":recall, 
            'f1score':f1score,
        } 

        return scores


    def get_stats_df(self, cm, dataset = "None"):

        scores = self.get_scores(cm)
        row_lim = self.row_lim
        expt_num = self.expt_num
        none_ratio = self.none_ratio

        recall = scores['recall']
        f1score = scores['f1score']
        precision = scores['precision']

        topic_stats_df = pd.DataFrame(self.topic_counts,columns=['Topic','Occurrences'])
        
        total_occurrences = topic_stats_df.Occurrences.sum()

        topic_stats_df['Proportion'] = topic_stats_df.Occurrences/total_occurrences
        topic_stats_df['Precision'] = precision.tolist()
        topic_stats_df['Recall'] = recall.tolist()
        topic_stats_df['F1score'] = f1score.tolist()

        overall_precision = (topic_stats_df.Precision * topic_stats_df.Proportion).sum()

        overall_recall = (topic_stats_df.Recall * topic_stats_df.Proportion).sum()

        overall_f1score = (topic_stats_df.F1score * topic_stats_df.Proportion).sum()
        
        topic_stats_df.loc['Overall'] = [
            'Overall',
            topic_stats_df.Occurrences.sum(),
            topic_stats_df.Proportion.sum(),
            overall_precision,
            overall_recall,
            overall_f1score]
        
        my_topics = ["Overall", 'None', 'abraham', 'passover', 'moses']
        
        selected_topics_df = topic_stats_df.loc[topic_stats_df['Topic'].isin(my_topics)]
        
        selected_topics_list = selected_topics_df['Topic'].to_list()

        selected_scores_list = selected_topics_df['F1score'].to_list()

        if self.should_print:
            # print(f'{dataset}')
            # print(topic_stats_df.round(2))
            print(f'\ntopics = ', selected_topics_list)
            print(f'{dataset}_expt_{expt_num} =', selected_scores_list)

        return topic_stats_df

class Trainer:
    
    def __init__(self, classifier):
    
        self.classifier = classifier


    def train(self, x_train, y_train):
        
        classifier = self.classifier
        
        try:
            classifier.fit(x_train, y_train)

        except:        
            y_train = y_train.values.toarray()
            
            classifier.fit(x_train, y_train)

        if isinstance(classifier, sklearn.model_selection._search.GridSearchCV):
            print ("Best params:",classifier.best_params_)
            print("Best score:",classifier.best_score_)
        
        return classifier

# # class Classifier
# # class Evaluator
