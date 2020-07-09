import pandas as pd
import warnings

# ignore warnings regarding column assignment, e.g. df['col1'] = list1 -- not 100% sure about this
pd.options.mode.chained_assignment = None  # default='warn'

# do not limit num of rows in df to display
pd.set_option('display.max_rows', None)

# disable future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# width of column to dispaly in dataframe
pd.options.display.max_colwidth = 40