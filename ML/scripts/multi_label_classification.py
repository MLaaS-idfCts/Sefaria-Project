import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score
from classes import DataManager, PipelineFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

NUM_TOPICS = 10
NUM_DATA_POINTS = 5000
DATA_PATH = '/persistent/Sefaria-Project/ML/data/yishai_data.csv'
MY_TOPICS = ['prayer', 'procedures-for-judges-and-conduct-towards-them', 'learning', 'kings', 'hilchot-chol-hamoed', 'laws-of-judges-and-courts', 'laws-of-animal-sacrifices', 'financial-ramifications-of-marriage', 'idolatry', 'laws-of-transferring-between-domains']

pd.options.display.max_colwidth = 50

df = pd.read_csv(DATA_PATH)[:NUM_DATA_POINTS]

# init data manager class
data = DataManager(
    raw = df, 
    num_topics = NUM_TOPICS, 
    my_topics = MY_TOPICS,
    should_clean = True,
    should_stem = True
    )

# result = data.preprocess_dataframe()
# result = data.get_topic_counts()
# result = data.show_topic_counts()
# result = data.show_simple_plot()
# result.show()

# result = data.clean_text()
# result = data.remove_stopwords()
result = data.stem_words()
# print(result.head(5))

data = result

# train test split
train, test = train_test_split(data, random_state=42, test_size=0.30, 
#                                shuffle=True
                              )
# print(train.head(5))
# print(test.head(5))

train_text = train['passage_text']
test_text = test['passage_text']

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
# vectorizer.fit(test_text)

x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['passage_text'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['passage_text'], axis=1)

# result = y_test.head()

def get_true_topics(row):
    true_topics = []
    global MY_TOPICS
    for col in MY_TOPICS:
        try:
           row[col]
        except:
            continue
        if row[col]==1:
            true_topics.append(col)
    return true_topics

test['true_topics'] = test.apply(get_true_topics,axis=1)

# using binary relevance
from skmultilearn.adapt import MLkNN
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain

classifier_type = None

classifier = OneVsRestClassifier(LogisticRegression())  # 1%
# classifier = LabelPowerset(LogisticRegression())        # 52%
# classifier, classifier_type = MLkNN(k=10), 'kNN'        # 43%
# classifier = BinaryRelevance(GaussianNB())              # 16%
# classifier = ClassifierChain(LogisticRegression())      # 14%

if classifier_type == 'kNN':
    x_train = lil_matrix(x_train).toarray()
    y_train = lil_matrix(y_train).toarray()
    x_test = lil_matrix(x_test).toarray()

# train
classifier.fit(x_train, y_train)

# predict"
test_predictions = classifier.predict(x_test)
train_predictions = classifier.predict(x_train)

my_example_topics = MY_TOPICS
preds_list = list(test_predictions)
preds_col = [
    [my_example_topics[int(s)] for s in (str(item)[:-2]) if s.isdigit()]
     for item in preds_list]
test['pred_topics'] = preds_col




result = test[['passage_text','true_topics','pred_topics']].head()
print(result)

# accuracy
print("Test Accuracy = ",accuracy_score(y_test,test_predictions))
print("Train Accuracy = ",accuracy_score(y_train,train_predictions))
sys.exit()

if False:
    # get most poular topics
    top_topics_df = data_manager.get_top_topics()
    top_topics_list = list(top_topics_df['topic']) # + ['ammon']
    print(top_topics_list)
    # top_topics = 'laws-of-judges-and-courts judgements1 laws-of-setting-the-months-and-leap-years sanhedrin'.split()
    # top_topics = 'ammon'.split()
    # top_topics = 'fate-of-the-nations-of-the-world punishment'.split()


# select a model: Linear SVC, Multinimial Naive-Bayes, or Logistic Regression
pipeline = PipelineFactory(
    'LinSVC'
    # 'LogReg'
    # 'MultNB' # seems buggy! predicts all zeroes!
    ).get_pipeline()
