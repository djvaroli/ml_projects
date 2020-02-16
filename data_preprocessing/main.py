# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# importing libraries 

import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling
from sklearn.compose import ColumnTransformer # create onehot encoder transform


dataset = pd.read_csv('DataPreprocessing.csv') # to import the dataset into a variable
X = dataset.iloc[:, :-1].values # attributes to determine dependent variable / Class
Y = dataset.iloc[:, -1].values # dependent variable / Class

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,1:])
X[:,1:] = imputer.transform(X[:, 1:])

labelencoder_X = LabelEncoder()
labelencoder_Y = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
Y = labelencoder_Y.fit_transform(Y)

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [0]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)


X = transformer.fit_transform(X)
X[:,0:3] = X[:,0:3].astype('int64') # convert onehot encoders to ints

#split the dataset into test and train subsets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)








