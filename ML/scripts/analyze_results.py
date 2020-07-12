import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

np.set_printoptions(linewidth=np.inf)


# to_analyze = 'scores'
# to_analyze = 'preds'
to_analyze = 'cm'


if to_analyze == 'scores':

    path = '/persistent/Sefaria-Project/ML/data/score_df_120000.pkl'

    score_df = pd.read_pickle(path)

    print(score_df)


if to_analyze == 'preds':

    path = '/persistent/Sefaria-Project/ML/data/pred_vs_true_80000.pkl'

    pred_vs_true = pd.read_pickle(path)

    print(pred_vs_true.sample(30))


if to_analyze == 'cm':

    path = '/persistent/Sefaria-Project/ML/data/cm_80000.dat'

    cm = np.load(path,allow_pickle=True)

    print(cm)


    # vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
    #                 "potato", "wheat", "barley"]
    # farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
    #             "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]


    # fig, ax = plt.subplots()
    
    # im = ax.imshow(cm)

    # # We want to show all ticks...
    # ax.set_xticks(np.arange(len(farmers)))
    # ax.set_yticks(np.arange(len(vegetables)))
    # # # ... and label them with the respective list entries
    # ax.set_xticklabels(farmers)
    # ax.set_yticklabels(vegetables)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(vegetables)):
    #     for j in range(len(farmers)):
    #         text = ax.text(j, i, cm[i, j],
    #                     ha="center", va="center", color="w")

    # ax.set_title("cm of local farmers (in tons/year)")
    # fig.tight_layout()
    # plt.show()