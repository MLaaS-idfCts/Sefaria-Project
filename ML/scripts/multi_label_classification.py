import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.svm import SVC
from datetime import datetime
from skmultilearn.adapt import MLkNN
from sklearn.pipeline import Pipeline
# from fastxml import Trainer, Inferencer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix, lil_matrix
from classes import DataManager, PipelineFactory
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', None)

NUM_TOPICS = 3
ROW_LIMIT = None
DATA_PATH = '/persistent/Sefaria-Project/ML/data/yishai_data.csv'

classifier_types = {
                                    # recall    precision
    1:'One Vs Rest',                # 25%       100%
    2:'Binary Relevance',           
    3:'Classifier Chains',          
    4:'Label Powerset',             
    5:'k Nearest Neighbors',        
}

# choose your classifier type
classifier_key = 2

classifier_type = classifier_types[classifier_key]

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

x_train = vectorizer.transform(train_text)
y_train = train.drop(labels = ['passage_text','true_topics'], axis=1)

x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['passage_text','true_topics'], axis=1)

if classifier_type == 'Binary Relevance':
    classifier = BinaryRelevance(GaussianNB())              # 16%

if classifier_type =='Classifier Chains':
    classifier = ClassifierChain(LogisticRegression())      # 14%

if classifier_type == 'Label Powerset':
    classifier = LabelPowerset(LogisticRegression())        # 52%

if classifier_type == 'k Nearest Neighbors':
    classifier = MLkNN(k=10)                                # 43%
    x_train = lil_matrix(x_train).toarray()
    y_train = lil_matrix(y_train).toarray()
    x_test = lil_matrix(x_test).toarray()

if classifier_type == 'One Vs Rest':    
    classifier = OneVsRestClassifier(LogisticRegression())
if False:
    LogReg_pipeline = Pipeline([('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1))])
    categories = top_topics
    for category in tqdm(categories):
        LogReg_pipeline.fit(x_train, train[category])
        prediction = LogReg_pipeline.predict(x_test)
        for row, value in enumerate(list(prediction)):
            if value != 0:
                if test['pred_topics'].iloc[row] is None:
                    test['pred_topics'].iloc[row] = [category]
                else:
                    test['pred_topics'].iloc[row] = test['pred_topics'].iloc[row].append(category)

# if classifier_type != 'One Vs Rest':
if True:    
    classifier.fit(x_train, y_train)
    test_predictions = classifier.predict(x_test)
    train_predictions = classifier.predict(x_train)
    preds_list = list(test_predictions)
    preds_col = [
        [top_topics[int(s)] for s in (str(item)[:-2]) if s.isdigit()]
        for item in preds_list]
    test['pred_topics'] = preds_col


cols=['passage_text','true_topics','pred_topics']
result = test[cols]
# result[''] = ''
# result = result.stack()
print(result)
print('\nClassifier type:',classifier_type)
# print("Accuracy = ",accuracy_score(y_test,test_predictions))

true_label_lists = result.true_topics.tolist()
pred_label_lists = result.pred_topics.tolist()



assert len(true_label_lists) == len(pred_label_lists)

num_passages = len(true_label_lists)

y_true = []
y_pred = []

for i in range(num_passages):

    true_label_list = true_label_lists[i]
    pred_label_list = pred_label_lists[i]

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
y_actu = pd.Series(y_true, name='Actual')
y_pred = pd.Series(y_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)
print()