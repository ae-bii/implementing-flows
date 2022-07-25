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
moon_1 = sklearn.datasets.make_moons(n_samples=500, shuffle=True, noise=0.05, random_state=None)
for i in range(0,200):
    data_1.append(moon_1[0][i][0])
data_1 = np.transpose(data_1)
# second moon sample:
moon_2 = sklearn.datasets.make_moons(n_samples=500, shuffle=True, noise=0.005, random_state=None)
for i in range(0,200):
    data_2.append(moon_2[0][i][0])
data_2 = np.transpose(data_2)
# third moon sample:
moon_3 = sklearn.datasets.make_moons(n_samples=500, shuffle=True, noise=1, random_state=None)
for i in range(0,200):
    data_3.append(moon_3[0][i][0])
data_3 = np.transpose(data_3)
# more moon samples...

DATA = {"1":data_1,
        "2":data_2,
        "3":data_3}
df = pd.DataFrame(DATA)
df.to_csv('3D_moon.csv', index = False)
