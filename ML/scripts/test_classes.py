import pandas as pd
from classes import DataManager

df = pd.read_csv(
    '/root/Sefaria-Project/ML/data/yishai_data.csv'
    )[:10000]
data_manager = DataManager(df)
result = data_manager._get_labeled()['labeled'].head()
print(result)
print("Finished!")