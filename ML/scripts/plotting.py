import sys
import matplotlib.pyplot as plt
import numpy as np

# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2*np.pi*t)
# plt.plot(t, s)

# plt.title('About as simple as it gets, folks')
# plt.show()


path = '/persistent/Sefaria-Project/ML/data/cm_80000.dat'

cm = np.load(path,allow_pickle=True)

subsize = 13

cm = cm[:subsize,:subsize]

print(cm)

# sys.exit()


# vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
#                 "potato", "wheat", "barley"]

farmers = [
    # 'OVERALL', '---', 
    'None', 
    'abraham', 'dinei-yibum', 
            'financial-ramifications-of-marriage', 'haggadah', 
            'idolatry', 'jacob', 'joseph', 'king-david', 'laws-of-judges-and-courts', 
            'laws-of-transferring-between-domains', 'leadership', 'learning', 
            # 'moses', 'passover', 'prayer', 'procedures-for-judges-and-conduct-towards-them', 
            # 'rabbinically-forbidden-activities-on-shabbat', 'teshuvah', 'torah', 
            # 'women'
            ]
vegetables = farmers
# farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#             "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]


fig, ax = plt.subplots(figsize=(20, 10))

im = ax.imshow(cm)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# # ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="w")

ax.set_title("cm of local farmers (in tons/year)")
fig.tight_layout()
# plt.figure(figsize=(20,10))
plt.show()