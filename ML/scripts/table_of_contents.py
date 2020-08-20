import os
import sys
import csv
import pickle
import django
import matplotlib.pyplot as plt

sys.path.insert(1, '/persistent/Sefaria-Project/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'sefaria.settings'
django.setup()

from sefaria.model import *
from sefaria.system.database import db

laws_str = """halachic-principles
human-ethics
family-law
laws-of-prayer
laws-of-kindness
property-law
tort-law
laws-of-impurity-and-purity
agricultural-law
laws-of-optional-restrictions
laws-of-scribes
laws-of-the-calendar
laws-of-worship-of-god
laws-of-government
laws-of-food
laws-of-clothing
noahide-(gentile)-law"""

laws_list = laws_str.split()

roots = TopicSet({"isTopLevelDisplay": True})

topic_grouping = {}

for root in roots:

    children = root.topics_by_link_type_recursively(linkType='displays-under')

    topic_grouping[root.slug] = sorted(list(set([child.slug for child in children])))

div_laws_options = ['laws_united','laws_divided']

for div_laws in div_laws_options:

    if div_laws == 'laws_united':

        pass

    if div_laws == 'laws_divided':

        copy = topic_grouping

        topic_grouping = {}

        for k,v in copy.items():

            if k != 'laws':

                topic_grouping[k] = v

        for law_topic in laws_list:

            children_obj_lst = Topic.init(law_topic).topics_by_link_type_recursively()

            children_names_list = [child_obj.slug for child_obj in children_obj_lst]

            topic_grouping[law_topic] = children_names_list
            
    path = f'data/topic_counts_{div_laws}.pickle'

    with open(path, 'wb') as handle:

        pickle.dump(topic_grouping, handle, protocol=3)

    topic_grouping = None

    with open(path, 'rb') as handle:

        topic_grouping = pickle.load(handle)

    contents_counts = {}

    for key, value in topic_grouping.items():

        contents_counts[key] = len(value)

    keys = contents_counts.keys()

    values = contents_counts.values()

    plt.bar(keys, values)

    plt.xticks(rotation=90)

    plt.title(f'Table of Contents:\n{div_laws.upper()}')

    plt.xlabel('Topic Group')

    plt.ylabel('Number of Child Topics')

    plt.savefig(f'images/topic_counts_{div_laws}.png', bbox_inches='tight')

    plt.clf()

print()