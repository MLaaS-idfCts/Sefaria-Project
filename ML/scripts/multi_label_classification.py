import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from sklearn.utils import shuffle
from skmultilearn.adapt import MLkNN
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
# from fastxml import Trainer, Inferencer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.datasets import make_classification
from classes import DataManager, PipelineFactory
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', None)

NUM_TOPICS = 5
ROW_LIMIT = 10000
DATA_PATH = '/persistent/Sefaria-Project/ML/data/yishai_data.csv'

classifier_types = {
    1:OneVsRestClassifier(SVC()),
    2:OneVsRestClassifier(LinearSVC()),
    # 3:OneVsRestClassifier(LogisticRegression()),
    # 4:OneVsRestClassifier(MultinomialNB()),
    # 5:OneVsRestClassifier(GaussianNB()),
    
    # 6:BinaryRelevance(SVC()),
    7:BinaryRelevance(LinearSVC()),
    8:BinaryRelevance(LogisticRegression()),
    9:BinaryRelevance(MultinomialNB()),
    10:BinaryRelevance(GaussianNB()),
    
    11:ClassifierChain(SVC()),
    12:ClassifierChain(LinearSVC()),
    13:ClassifierChain(LogisticRegression()),
    14:ClassifierChain(MultinomialNB()),
    15:ClassifierChain(GaussianNB()),
    
    16:LabelPowerset(SVC()),
    17:LabelPowerset(LinearSVC()),
    18:LabelPowerset(LogisticRegression()),
    19:LabelPowerset(MultinomialNB()),
    20:LabelPowerset(GaussianNB()),
    
}

pd.options.display.max_colwidth = 50

df = pd.read_csv(DATA_PATH)[:ROW_LIMIT]

data = DataManager(raw_df = df, num_topics = NUM_TOPICS, should_clean = True, should_stem = False)

top_topics = data._get_top_topics()

result = data.preprocess_dataframe()

data = result

train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)

test['pred_topics'] = None

train_text = train['passage_text']
test_text = test['passage_text']

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
# only fit with training data, not with testing data, because testing data should be "UNSEEN"
vectorizer.fit(train_text)

# numerical version of words in passage
x_train = vectorizer.transform(train_text)
# topics columns with 1 in row if that passage belongs
y_train = train.drop(labels = ['passage_text','true_topics'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['passage_text','true_topics'], axis=1)

for classifier_key in classifier_types.keys():
    # choose your classifier type

    classifier = classifier_types[classifier_key]

    classifier.fit(x_train, y_train)
    test_predictions = classifier.predict(x_test)
    train_predictions = classifier.predict(x_train)

    preds_list = list(test_predictions)
    pred_labels_list = []
    for array in preds_list:
        passage_labels = []
        for topic_index, pred_value in enumerate(array):
            if pred_value != 0:
                passage_labels.append(top_topics[topic_index])
        pred_labels_list.append(passage_labels)

    test['pred_topics'] = pred_labels_list

    cols=['passage_text','true_topics','pred_topics']

    result = test[cols]
    topics_comparison = test[['true_topics','pred_topics']]
    # print(topics_comparison)
    print('\nClassifier type:',classifier)
    print('**********************************************')

    topics_list = ['None'] + top_topics

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
            
    import pandas as pd

    y_actu = pd.Categorical(y_true, categories=topics_list)
    y_pred = pd.Categorical(y_pred, categories=topics_list)

    cm = pd.crosstab(y_actu, y_pred, rownames=['True'], colnames = ['Pred'], dropna=False) 
    # print(cm)
    cm = cm.values

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    
    np.seterr(divide='ignore', invalid='ignore')

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)


    precision_dict = dict(zip(topics_list, precision))

    recall_dict = dict(zip(topics_list, recall))
    # print()
    print('Precision:',[round(num,2) for num in precision])
    # for k,v in precision_dict.items():
    #     print(k)
    #     print(v)

    print('Recall:',[round(num,2) for num in recall])
    # for k,v in recall_dict.items():
    #     print(k)
        # print(v)

print([(topic_index, topic) for topic_index, topic in enumerate(top_topics)])
print()