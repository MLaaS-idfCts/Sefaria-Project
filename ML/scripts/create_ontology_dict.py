import os
import sys
import csv
import pickle
import django

sys.path.insert(1, '/persistent/Sefaria-Project/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'sefaria.settings'
django.setup()

from tqdm import tqdm
from sefaria.model import *
from sefaria.system.database import db

# from conquer_ontology import get_last_leaves



def get_children(slug):
    slug.replace('--','-')
    # if slug == None:
    #     return None
    topic = Topic.init(slug)
    try:
        children_links = topic.link_set(query_kwargs={"linkType": 'is-a', 'toTopic': slug})
    except:
        return []
    # try:
    #     children_links = topic.link_set(query_kwargs={"linkType": 'is-a', 'toTopic': slug})
    # except:
    #     return None
    children_slugs = [child.topic for child in children_links]
    return children_slugs


threshold = 40

def get_ontology_dict(node):
    
    result = {}

    children = get_children(node)

    if children == [] :
        return None

    else:
        for child in children:
            result[child] = get_ontology_dict(child)
    
    return result

    
# my_node = 'musical-process'
# my_node = 'art'
my_node = 'role'
# my_node = 'kings'

# for k,v in get_ontology_dict(my_node).items():
    # print(k,v)


# entire_ontology_dict = get_ontology_dict('art')
# entire_ontology_dict = get_ontology_dict('entity')


with open(f'data/entire_ontology_dict.pickle', 'wb') as handle:
    pickle.dump(entire_ontology_dict, handle, protocol=3)

# path = 'data/entire_ontology_dict.pickle'
# with open(path, 'rb') as handle:

#     entire_ontology_dict_recalled = pickle.load(handle)

# is_same = entire_ontology_dict_recalled == entire_ontology_dict
# print(is_same)
print()


