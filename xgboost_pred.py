from IPython.core.debugger import set_trace

# %load_ext nb_black

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from xgboost import XGBRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
model = XGBRegressor()
model.load_model("xgboost_model.txt")

plt.style.use(style="seaborn")
# %matplotlib inline

df = pd.read_csv("./stock_data.csv")
df.head(5)
df = df[["Close"]].copy()
df.head(5)

def train_test_split_value(data, percent):
    data = data.values
    n = int(len(data) * (1 - percent))
    return data[:n], data[n:]

train, test = train_test_split_value(df, 0.2)    

X = train[:, :-1]
y = train[:, -1]

val = np.array(test[0, 0]).reshape(1, -1)
pred = model.predict(val)
print(pred[0])