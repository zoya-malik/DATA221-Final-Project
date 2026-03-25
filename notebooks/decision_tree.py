import pandas as pd

#Load data
df = pd.read_csv('../data/cleaned_data.csv')
y = df['TARGET']
X = df.drop(columns=['TARGET'])
X = pd.get_dummies(X)
