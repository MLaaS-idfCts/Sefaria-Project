
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