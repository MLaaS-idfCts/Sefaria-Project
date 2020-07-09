import pandas as pd

from multi_label_classification import predictors, vectorizer


example_passages = [
    'king david king david king david king david king david king david king david king david king david king david king david.',
    'animal sacrifices animal sacrifices animal sacrifices animal sacrifices animal sacrifices animal sacrifices '
]

x_example_series = pd.Series(example_passages)

x_example = vectorizer.transform(x_example_series)

for predictor in predictors:

    print(predictor.get_preds_list(x_example))

print()