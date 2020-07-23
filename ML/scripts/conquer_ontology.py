import sys
import pickle 
import matplotlib.pyplot as plt
"""
inputs:
dict of counts of topics
- to obtain this, run count on expanded topics columns
- count how many times each topci eappear in that column
some threshold

outputs:
list of topics that fulfill threshold

breakdown to smaller problem:
don't run on all ontology
"mocking" -- focus on concpt

"""

ontology = {
  "a": {
    "b": {
      "c": {
        "d": None
      },
      "e": None,
      "f": None,
      "g": {
        "h": {
          "i": None
        }
      }
    },
    "j": {
      "k": None,
      "l": {
        "m": {
          "n": None,
          "o": None
        },
        "p": {
          "q": None,
          "r": None,
          "s": {
            "t": None,
            "u": None
          }
        }
      },
      "v": {
        "w": {
          "x": {
            "y": {
              "z": None
            }
          }
        }
      }
    }
  }
}

counts = {
  "z": 50,
  "y": 50,
  "x": 50,
  "w": 50,
  "v": 50,
  "u": 1400,
  "t": 67,
  "s": 1467,
  "r": 4,
  "q": 0,
  "p": 1471,
  "o": 34,
  "n": 99,
  "m": 133,
  "l": 1604,
  "k": 234,
  "j": 1888,
  "i": 78,
  "h": 78,
  "g": 78,
  "f": 24,
  "e": 10,
  "d": 100,
  "c": 100,
  "b": 212,
  "a": 2100
}
# input:

#   counts,
#   root,
#   threshold
# output:


def get_last_leaves(parent,children,threshold):
	
	# base cases
	if children == None:
		return {parent}

	for key in children.keys():
		if counts[key] < threshold:
			return {parent}

	# recursion case
	last_leaves = []

	for child, grandchildren in children.items():
		for leaf in get_last_leaves(child, grandchildren, threshold):
			last_leaves.append(leaf)
	
	leaf_set = set(last_leaves)

	return leaf_set
	
test_output = {"b", "z", "k", "m", "p"}

root = ontology['a']

threshold = 50

parent = 'a'

counts = None

import os
directory = 'data/ontology_counts'
for file_name in os.listdir(directory):
	path = os.path.join(directory,file_name)
	with open(path, 'rb') as handle:

		counts = pickle.load(handle)

	plt.xlabel('X = Number of children')
	plt.ylabel('Y = Number of nodes with X children')
	plt.xlim(1, 60)
	plt.ylim(0, 600)

	plt.hist( counts.values(), bins=max(counts.values()), width = 15.0, color='g')
	plt.savefig(f'images/{file_name[:-7]}.png')
	# continue

import matplotlib.pyplot as plt
plt.hist( counts.values(), bins=max(counts.values()), width = 10, color='g')
plt.show()

children = ontology[parent]

my_output = get_last_leaves(parent, children, threshold)

print("Does the test output match my output?", my_output == test_output)






def get_children(slug):
    topic = Topic.init(slug)
    children_links = topic.link_set(query_kwargs={"linkType": 'is-a', 'toTopic': slug})
    children_slugs = [child.topic for child in children_links]
    return children_slugs

threshold = 50

parent = 'entity'

children = get_children[parent]

my_output = get_last_leaves(parent, children, threshold)