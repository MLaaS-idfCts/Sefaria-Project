import sklearn
import re

class DataManager:

    def __init__(self, raw):
        self.raw = raw

    def is_labeled(item):
        result = None
        topics = item["Topics"]
        if topics:
            result = topics
        return result

    def is_labeled_split(self, raw):
        return {
            'labeled':[item for item in raw if is_labeled(item)],
            'unlabeled':[item for item in raw if not is_labeled(item)]
        }

    def train_test_split(self,labeled):
        return sklearn.model_selection.train_test_split(labeled)

    def clean_text(sentence):

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

    def clean(self,df_col):
        return df_col.apply(clean_text)

class Trainer
class Classifier
class Evaluator
