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

rows = []
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
topic_cache = {}
parent_cache = {}
for passage in tqdm(passage_list):
    try:
        ref = Ref(passage['_id'])
    except:
        print(f"Problem reading Ref(passage['_id']) for this passage --> {passage}.")
        continue
    topics = []
    for slug in passage['topics']:
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
    rows += [{
        "Ref": passage['_id'],
        "En": ref.text('en').as_string(),
        "He": ref.text('he').as_string(),
        "Topics": " ".join(passage['topics']),
        "Expanded Topics": " ".join(expanded_topics)
    }]
with open("yishai_data.csv", "w") as fout:
    c = csv.DictWriter(fout, ["Ref", "En", "He", "Topics", "Expanded Topics"])
    c.writeheader()
    c.writerows(rows)