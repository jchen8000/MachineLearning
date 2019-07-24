import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
df_iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                       columns= iris['feature_names'] + ['target'])


X = iris.data
y = iris.target

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

df_iris_train = pd.DataFrame(data= np.c_[X_train, y_train],
                       		 columns= iris['feature_names'] + ['target'])

df_iris_test = pd.DataFrame(data= np.c_[X_test, y_test],
                       		 columns= iris['feature_names'] + ['target'])