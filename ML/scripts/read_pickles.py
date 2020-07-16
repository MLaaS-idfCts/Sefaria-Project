import pandas as pd
path = '/persistent/Sefaria-Project/ML/data/test_score_df_0.6.pkl'
df = pd.read_pickle(path)
print(df)