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
        - add paresed_Ref column to show just relevant subject
            - e.g. "Mishna Torah, Shabbat, 4:7" --> "shabbat"
        - clean En column

    - convert topics list into one-hot-encoded columns

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
        return df.dropna(axis=0, subset=[['Ref', 'En']])

    def _remove_duplicates(self):
        df = self._remove_null()
        return df.drop_duplicates()

    def _clean_text(sentence):

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
        df = self._remove_duplicates()
        df.Ref = self._clean_text(df.Ref)
        df.En = self._clean_text(df.En)
        return df

    def _labeled_unlabeled_split(self):
        df = self._clean_columns()
        return {
            'labeled':df[df.Topics.notnull()],
            'unlabeled':df[df.Topics.isnull()]
        }

    def train_test_split(self,labeled):
        labeled_data = self._labeled_unlabeled_split()['labeled']
        return sklearn.model_selection.train_test_split(labeled_data)

class Trainer
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


class Classifier
class Evaluator
