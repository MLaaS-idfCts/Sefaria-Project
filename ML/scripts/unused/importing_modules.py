import sys
import time
import scipy
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm
from classes import DataManager
from datetime import datetime
from sklearn.svm import SVC, LinearSVC
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.adapt import BRkNNaClassifier, MLkNN
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from sklearn.feature_extraction.text import TfidfVectorizer

import re
from nltk import *
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
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
