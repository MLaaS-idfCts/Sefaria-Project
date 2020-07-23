import pandas as pd

s = pd.Series([0, 1])
ax = s.hist()  # s is an instance of Series
fig = ax.get_figure()
fig.savefig('example_plot.png')