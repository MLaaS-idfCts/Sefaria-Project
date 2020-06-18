from time import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from classes import DataManager, CustomPipeline

NUM_DATA_POINTS = 10000
NUM_TOPICS = 50

# import data
df = pd.read_csv('/root/Sefaria-Project/ML/data/yishai_data.csv')[:NUM_DATA_POINTS]

# init data manager
data_manager = DataManager(raw = df, num_topics = NUM_TOPICS)

# split train and test data
train,test = data_manager.train_test_split()

# select relevant input, e.g. words in passage
X_train = train.En
X_test = test.En

# get shape
print('training passages',X_train.shape[0])
print('testing passages',X_test.shape[0])

# get most poular topics
topic_stats = data_manager._topic_stats()
top_topics = list(topic_stats['topic'])


# select a model: Linear SVC, Multinimial Naive-Bayes, or Logistic Regression
pipeline = CustomPipeline(
    # 'LinSVC'
    # 'LogReg'
    'MultNB'
    )._get_pipeline()


# init
topic_accuracies = {}

# for each topic, train (i.e. "fit") and classify ("predict") and evaulate
for topic in tqdm(top_topics):
# for topic in top_topics:
    
    print(f'Processing {topic}')
    
    # train the model 
    pipeline.fit(X_train, train[topic])

    # predict 
    prediction = pipeline.predict(X_test)

    # record testing accuracy
    accuracy = accuracy_score(test[topic], prediction)
    topic_accuracies[topic] = round(accuracy,3)

# ranked_topic_accuracies = 
for topic, accuracy in topic_accuracies.items():
    print(accuracy, '<--', topic)

print(f"Finished at {time.now}!")