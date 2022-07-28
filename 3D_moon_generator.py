import numpy as np
import sklearn
from sklearn import datasets
import csv
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import transpose

data_1 = []
data_2 = []
data_3 = []

# first moon sample:
moon_1 = sklearn.datasets.make_moons(n_samples=1000, shuffle=False, noise=0.05, random_state=None)
for i in range(0,1000):
    data_1.append(moon_1[0][i][0])
    data_2.append(moon_1[0][i][1])

moon_1_duplicate = sklearn.datasets.make_moons(n_samples=1000, shuffle=True, noise=0.051, random_state=None)
for i in range(0,1000):
    data_3.append(moon_1_duplicate[0][i][0])

DATA = {"1":data_1,
        "2":data_2,
        "3":data_3}
df = pd.DataFrame(DATA)
df.to_csv('3D_moon.csv', header=False, index = False)
