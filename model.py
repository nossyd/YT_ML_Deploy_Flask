# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Read csv file
df = pd.read_csv('hiring.csv')

# Fill NaNs
df['experience'].fillna(0, inplace=True)
df['test_score'].fillna(df['test_score'].mean(), inplace=True)

# Features
X = df.iloc[:, :3]

# Converting words to int fcn
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

# Apply words to int fcn on experience dimension
X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

# Dependent Variable
y = df.iloc[:, -1]

## Model

# Note: Won't be splitting train test set due to small dataset thus will train model on all available data
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# Fit model on train
reg.fit(X, y)

# Saving model to disk
pickle.dump(reg, open('model.pkl', 'wb'))

# Load model to compare results + Test
#model = pickle.load(open('model.pkl', 'rb'))
#print(model.predict([[4, 6, 8]]))