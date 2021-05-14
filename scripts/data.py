import numpy as np
import matplotlib.pyplot as plt


path = "/data/tmp/Sefaria-Project/ML/data/sample_df.pickle"
plt.bar(np.load(path, allow_pickle=True)["Topic"][:-1],
         np.load(path, allow_pickle=True)["Occurrences"][:-1])
plt.xticks(rotation=90)
plt.show() 