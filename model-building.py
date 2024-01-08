import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

data = pd.read_csv('datasets/penguins_cleaned.csv')

y = data['species'].values
x_cat = data[['sex','island']].values
x_num = data.drop(columns=['species','sex','island']).values

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(x_cat)
x_cat = ohe.transform(x_cat)

x = np.hstack([x_cat,x_num])

clf = RandomForestClassifier()
clf.fit(x,y)

with open('models/clf.pkl','wb') as f:
    pickle.dump(clf,f)

with open('models/encoder.pkl','wb') as f:
    pickle.dump(ohe,f)