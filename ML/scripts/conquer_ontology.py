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

# ontology = {
#   "a": {
#     "b": {
#       "c": {
#         "d": None
#       },
#       "e": None,
#       "f": None,
#       "g": {
#         "h": {
#           "i": None
#         }
#       }
#     },
#     "j": {
#       "k": None,
#       "l": {
#         "m": {
#           "n": None,
#           "o": None
#         },
#         "p": {
#           "q": None,
#           "r": None,
#           "s": {
#             "t": None,
#             "u": None
#           }
#         }
#       },
#       "v": {
#         "w": {
#           "x": {
#             "y": {
#               "z": None
#             }
#           }
#         }
#       }
#     }
#   }
# }

ontology_path = 'data/entire_ontology_dict.pickle'
with open(ontology_path, 'rb') as handle:
	ontology = pickle.load(handle)


# counts = {
#   "z": 50,
#   "y": 50,
#   "x": 50,
#   "w": 50,
#   "v": 50,
#   "u": 1400,
#   "t": 67,
#   "s": 1467,
#   "r": 4,
#   "q": 0,
#   "p": 1471,
#   "o": 34,
#   "n": 99,
#   "m": 133,
#   "l": 1604,
#   "k": 234,
#   "j": 1888,
#   "i": 78,
#   "h": 78,
#   "g": 78,
#   "f": 24,
#   "e": 10,
#   "d": 100,
#   "c": 100,
#   "b": 212,
#   "a": 2100
# }

counts_path = 'data\ontology_counts\ontology_counts_dict_row_lim_170000.pickle'
# counts_path = 'data\ontology_counts\ontology_counts_dict_row_lim_100000.pickle'
# counts_path = 'data\ontology_counts\ontology_counts_dict_row_lim_10000.pickle'
with open(counts_path, 'rb') as handle:
	counts = pickle.load(handle)

counts = {key:val for key, val in counts.items() if val >= 10}


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
		
		if key not in counts:
			return {parent}

		if counts[key] < threshold: 
			return {parent}

	# recursion case
	last_leaves = []

	for child, grandchildren in children.items():
		for leaf in get_last_leaves(child, grandchildren, threshold):
			last_leaves.append(leaf)
	
	leaf_set = last_leaves
	# leaf_set = set(last_leaves)

	return leaf_set


# test_output = {"b", "z", "k", "m", "p"}

if __name__ == '__main__':

	thresholds = [16000, 16500, 17000, 17500, 18000]
	# thresholds = [40]

	# parent = 'a'
	parent = 'entity'

	children = ontology

	for threshold in thresholds:

		my_output = get_last_leaves(parent, children, threshold)
		print()
		print("threshold:", threshold)
		print()

		my_output.sort()
		print(my_output)

	print()
	# print("Does the test output match my output?", my_output == test_output)
