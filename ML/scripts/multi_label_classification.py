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
NUM_DATA_POINTS = 50000
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

# test['true_topics'] = test.apply(get_true_topics,axis=1)
# result = test[['passage_text','true_topics']].head()

# print(result)

# %%time

# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(x_train, y_train)

# predict"
test_predictions = classifier.predict(x_test)
train_predictions = classifier.predict(x_train)


# accuracy
print("Test Accuracy = ",accuracy_score(y_test,test_predictions))
print("Train Accuracy = ",accuracy_score(y_train,train_predictions))
print("\n")

sys.exit()

# split train and test data
train, test = data_manager.get_train_and_test()

# select relevant input, e.g. words in passage
X_train = train.En
X_test = test.En

MY_INDEX_LIST = range(2)
for MY_INDEX in MY_INDEX_LIST:
    print('\nACTUAL PASSAGE:',X_test.iloc[MY_INDEX])
    print('\nACTUAL TOPICS:',
    (test.iloc[MY_INDEX] == 1).idxmax(axis=1)
    )


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
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag',max_iter=500), n_jobs=-1)),
            ])

categories = my_example_topics

for category in categories:
    print('\n**Processing {} comments...**'.format(category))
    
    # Training logistic regression model on train data
    LogReg_pipeline.fit(x_train, train[category])
    
    # calculating test accuracy
    prediction = LogReg_pipeline.predict(x_test)
    print(f'Test accuracy is {accuracy_score(test[category], prediction)}')
    
    for i in range(15):
        print('\ntext:',test['En'].iloc[i])
        print('prediction:',prediction[i])
        print('actual label:',test[category].iloc[i])
    
finish_now(start_time)


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