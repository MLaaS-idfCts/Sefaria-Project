from datetime import datetime
start_time = datetime.now()

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from classes import DataManager, PipelineFactory

print(f'imports took {datetime.now() - start_time}')

start_time = datetime.now()
# stuff
print(f'stuff took {datetime.now() - start_time}')

NUM_DATA_POINTS = 100
NUM_TOPICS = 5

# import data
df = pd.read_csv('/root/Sefaria-Project/ML/data/yishai_data.csv')[:NUM_DATA_POINTS]

# init data manager
data_manager = DataManager(raw = df, num_topics = NUM_TOPICS)

# split train and test data
train, test = data_manager.train_test_split()

# select relevant input, e.g. words in passage
X_train = train.En
X_test = test.En

# get shape
print('training passages',X_train.shape[0])
print('testing passages',X_test.shape[0])

# get most poular topics
topic_stats = data_manager.topic_stats()
top_topics = list(topic_stats['topic'])


# select a model: Linear SVC, Multinimial Naive-Bayes, or Logistic Regression
pipeline = PipelineFactory(
    # 'LinSVC'
    # 'LogReg'
    'MultNB'
    ).get_pipeline()


# init
topic_accuracies = {}
train_topic_accuracies = {}

# for each topic, train (i.e. "fit") and classify ("predict") and evaulate
print(f'For each topic, the model is: training, predicting, and evaluating.')
for topic in tqdm(top_topics):
# for topic in top_topics:
    
    # train the model 
    pipeline.fit(X_train, train[topic])

    # make predictions
    train_prediction = pipeline.predict(X_train)
    prediction = pipeline.predict(X_test)

    # evaluate and keep record of performance
    train_accuracy = accuracy_score(train[topic], train_prediction)
    train_topic_accuracies[topic] = round(train_accuracy,3)
    
    accuracy = accuracy_score(test[topic], prediction)
    topic_accuracies[topic] = round(accuracy,3)

# ranked_topic_accuracies = 
for topic in top_topics:
# for topic, accuracy in topic_accuracies.items():
    print(topic)
    print(train_topic_accuracies[topic], '<--', "train")
    print(topic_accuracies[topic], '<--', "test")

# classifier.fit(X_train, y_train)
# y_score = classifier.decision_function(X_test)
# Compute the average precision score
# from sklearn.metrics import average_precision_score
# average_precision = average_precision_score(y_test, y_score)

# print('Average precision-recall score: {0:0.2f}'.format(
#       average_precision))


# from sklearn.metrics import confusion_matrix

# conf_mat = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(10,10))
# sns.heatmap(conf_mat, annot=True, fmt='d',
#             xticklabels=category_id_df.Product.values, yticklabels=category_id_df.Product.values)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

print(test.iloc[0,1])

selected_topics = []
selected_topics = ["some topic"]
# selected_topics = [topic for topic in test.columns[4:] if test[:,topic] != 0]
for idx, selected_topic in enumerate(test.columns[:10]):
# for selected_topic in selected_topics:
    print(idx, selected_topic)
print(f"Finished at {datetime.now()} for {NUM_DATA_POINTS} rows and {NUM_TOPICS} topics!")