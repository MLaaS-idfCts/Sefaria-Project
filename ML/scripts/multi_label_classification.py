import sys

import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score
from classes import DataManager, PipelineFactory, my_example_topics


NUM_TOPICS = 20
NUM_DATA_POINTS = 1000
pd.options.display.max_colwidth = 50

# import data
# df = pd.read_csv('/root/Sefaria-Project/ML/data/yishai_data.csv')[:NUM_DATA_POINTS]
df = pd.read_pickle('/root/Sefaria-Project/ML/data/1k.pkl')
# df.to_pickle('/root/Sefaria-Project/ML/data/1k.pkl')
# df.set_index('Ref',
#     drop=False,
#     inplace=True)

# init data manager class
data_manager = DataManager(raw = df, num_topics = NUM_TOPICS)

# split train and test data
train, test = data_manager.get_train_and_test()

# select relevant input, e.g. words in passage
X_train = train.En
X_test = test.En

# MY_INDEX = 1
MY_INDEX_LIST = range(5)
for MY_INDEX in MY_INDEX_LIST:
    print('\nACTUAL PASSAGE:',X_test.iloc[MY_INDEX])
    print('\nACTUAL TOPICS:',
    # test.iloc[MY_INDEX]
    # test.columns[(test == 1).iloc[MY_INDEX]]
    (test.iloc[MY_INDEX] == 1).idxmax(axis=1)
    )

sys.exit()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(X_train)
# vectorizer.fit(test_text)

x_train = vectorizer.transform(X_train)
y_train = train.drop(labels = ['Ref','En','Topics'], axis=1)

x_test = vectorizer.transform(X_test)
y_test = test.drop(labels = ['Ref','En','Topics'], axis=1)

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
            ])

categories = my_example_topics


for category in categories:
    print('\n**Processing {} comments...**'.format(category))
    
    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, train[category])
    
    # calculating test accuracy
    prediction = LogReg_pipeline.predict(x_test)
    print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))

sys.exit()

# get shape
print('\ntraining passages',X_train.shape[0])
print('testing passages',X_test.shape[0])

# get most poular topics
top_topics_df = data_manager.get_top_topics()
top_topics_list = list(top_topics_df['topic']) # + ['ammon']
print(top_topics_list)
# top_topics = 'laws-of-judges-and-courts judgements1 laws-of-setting-the-months-and-leap-years sanhedrin'.split()
# top_topics = 'ammon'.split()
# top_topics = 'fate-of-the-nations-of-the-world punishment'.split()


# select a model: Linear SVC, Multinimial Naive-Bayes, or Logistic Regression
pipeline = PipelineFactory(
    # 'LinSVC'
    'LogReg'
    # 'MultNB' # seems buggy! predicts all zeroes!
    ).get_pipeline()


# init
topic_accuracies_testing = {}
topic_accuracies_training = {}

# for each topic, train (i.e. "fit") and classify ("predict") and evaulate
print(f'For each topic, the model is: training, predicting, and evaluating.')
# for topic in tqdm(top_topics_list):
for topic in top_topics_list:
    
    # train the model 
    pipeline.fit(X_train, train[topic])

    # make predictions
    prediction_training = pipeline.predict(X_train)
    prediction_testing = pipeline.predict(X_test)

    for i in range(test.shape[0]):
        if prediction_testing[i] != 0:

    # for i in range(train.shape[0]):
    #     if prediction_training[i] != 0:

            print(f"{topic} --> for test item #{i}!")
            # continue

    # print(prediction_testing[test_index].shape)
    # print(type(prediction_testing))
    # print()
    # my_prediction = pipeline.predict(X_test[test_index:test_index+1])

    # evaluate and record performance
    train_accuracy = accuracy_score(train[topic], prediction_training)
    topic_accuracies_training[topic] = round(train_accuracy,3)
    
    test_accuracy = accuracy_score(test[topic], prediction_testing)
    topic_accuracies_testing[topic] = round(test_accuracy,3)

# ranked_topic_accuracies = 
for topic in top_topics_list:
# for topic, accuracy in topic_accuracies.items():
    # if True:
    if False:
        print()
        print(topic)
        print(topic_accuracies_training[topic], '<--', "train")
        print(topic_accuracies_testing[topic], '<--', "test")
    continue



selected_topics = []
# selected_topics = ["some topic"]

# selected_topics = [topic for topic in test.columns[4:] if test[:,topic] != 0]
# for idx, selected_topic in enumerate(test.columns[:10]):

for idx, selected_topic in enumerate(selected_topics):
    print(idx, selected_topic)

print(f"\nFinished at {datetime.now()} for {NUM_DATA_POINTS} rows and {NUM_TOPICS} topics!")