import django
django.setup()

from sefaria.model import *
import sefaria.system.database as database

from collections import defaultdict

talmudic_topic_hits = {}

my_ref = Ref("Leviticus 19:18")

for link in my_ref.linkset():

    # determine which link is the "other" one
    side = 0
    if my_ref.normal() in set(getattr(link, 'expandedRefs0')):
        side = 1
    other_ref = Ref(link.refs[side])

    # exclude topics from anything but talmud
    if other_ref.index.categories[0] != "Talmud":
        continue
    topic_links = other_ref.topiclinkset()

    # incr value of this topic in the dict
    for topic_link in topic_links:
        topic = topic_link.toTopic
        if topic not in talmudic_topic_hits:
            talmudic_topic_hits[topic] = 1
        else:
            talmudic_topic_hits[topic] += 1

ranked_talmudic_topics = sorted(talmudic_topic_hits.items(), key=lambda x: x[1], reverse=True)

for topic in ranked_talmudic_topics[:10]:
    print(topic)