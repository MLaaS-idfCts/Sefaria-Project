import os
import sys
import csv
import django

sys.path.insert(1, '/persistent/Sefaria-Project/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'sefaria.settings'
django.setup()

from tqdm import tqdm
from sefaria.model import *
from sefaria.system.database import db

aggregated_passages = db.topic_links.aggregate(
    [
        { 
            "$match" : { 
                "is_sheet" : False, 
                "class" : "refTopic"
            }
        }, 
        { 
            "$unwind" : { 
                "path" : "$expandedRefs"
            }
        }, 
        { 
            "$group" : { 
                "_id" : "$expandedRefs", 
                "topics" : { 
                    "$addToSet" : "$toTopic"
                }
            }
        }
    ]
);

passage_list = list(aggregated_passages)

rows = []

topic_cache = {}

parent_cache = {}

# *****************
# passage_limit = 100
passage_limit = None
# *****************

for passage_info in tqdm(passage_list[:passage_limit]):

    # sometimes it is problematic to read the reference
    try:
        ref = Ref(passage_info['_id'])
 
    # print error, skip this passage, and move on to the next passage
    except:
        print(f"Problem reading Ref(passage['_id']) for this passage --> {passage_info['_id']}.")
        continue

    try:
        hebrew_prefixed_text = db.prefixes.find_one({"ref": ref.orig_tref})['text']    
    except:
        print(f"Problem reading dicta prefixed version of hebrew text this passage_info --> {passage_info['_id']}.")
        continue

    # init list of version titles for various english translations
    version_titles = []

    # each item in list is a dict of info about that version
    version_list = ref.version_list()
    
    # select those which meet our criteria
    for version_info in version_list:
    
        # gather info needed, title of version and language of characters 
        version_title = version_info['versionTitle']
        # this is not necessarily language of words, e.g. german or spanish will also appear as 'en'
        version_language = version_info['language']
    
        # store whether info is relevant
        has_eng_chars = version_language == 'en' 
        is_foreign_lang = version_title[-5:-3] == ' [' # e.g. if is german, there will be a '[de]' at the end.

        # if relevent, include that title
        if has_eng_chars and not is_foreign_lang:

            version_titles.append(version_title)

    # init
    topics = []
    
    for slug in passage_info['topics']:

        if slug in topic_cache:
            topic = topic_cache[slug]
        
        else:
            topic = Topic.init(slug)
            if not topic:
                print("Topic is None:", slug)
                continue
                topic_cache[slug] = topic
        
        topics += [topic]
    
    expanded_topics = set()
    
    for topic in topics:
        if topic.slug in parent_cache:
            parents = parent_cache[topic.slug]
        else:    
            parents = {parent.slug for parent in topic.topics_by_link_type_recursively(reverse=True)}
            parent_cache[topic.slug] = parents
        expanded_topics |= parents
    
    # init, to eventually concatenate all versions' texts
    english_texts_lst = []

    for version_title in version_titles:
            
        english_text = ref.text('en',version_title).as_string()

        english_texts_lst.append(english_text)
    
    english_texts_str = ' '.join(english_texts_lst)

    hebrew_text_str = ref.text('he').as_string()

    rows += [{
        "Ref": passage_info['_id'],
        "En": english_texts_str,
        "He": hebrew_text_str,
        "He_prefixed": hebrew_prefixed_text,
        "Topics": " ".join(passage_info['topics']),
        "Expanded Topics": " ".join(expanded_topics)
    }]

with open("/persistent/Sefaria-Project/ML/data/concat_english_prefix_hebrew.csv", "w") as fout:
    c = csv.DictWriter(fout, ["Ref", "En", "He", "He_prefixed", "Topics", "Expanded Topics"])
    c.writeheader()
    c.writerows(rows)