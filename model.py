import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('data.csv')

X = df.columns.drop('Outcome')
y = 'Outcome'

clf = LogisticRegression(random_state=0)
clf.fit(df[X], df[y])

joblib.dump(clf, "clf.pkl")