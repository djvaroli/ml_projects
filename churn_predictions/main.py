import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sklearn as sk
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense



try:
    basedir = os.path.abspath(os.path.dirname(__file__))
    datafile = os.path.join(basedir, 'Churn_Modelling.csv')
except:
    datafile = 'Churn_Modelling.csv'
    
dataset = pd.read_csv(datafile)

# remove name, lastname and row number from our dataset
X = dataset.iloc[:, 3:13]
# store the dependent variable in y, i.e. whether the customer left or not
y = dataset.iloc[:, 13].values


geo_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
X.Geography = geo_encoder.fit_transform(X.Geography) # encode
X.Gender = gender_encoder.fit_transform(X.Gender)

transformer = ColumnTransformer(
    transformers=[
        ('OneHotEncoder',
         OneHotEncoder(),
         [1]
        )
    ],
    remainder="passthrough"
)

X = transformer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#initialize the classifier and add dense layers

classifier = Sequential()
classifier.add(Dense(output_dim=6, init='uniform',
                     activation='relu', input_dim=12))
classifier.add(Dense(output_dim=6, init='uniform',
                     activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', 
                     activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

classifier.fit(X_train, Y_train, batch_size=10, nb_epoch=100)


y_pred = classifier.predict(X_test) > 0.5

cm = confusion_matrix(Y_test, y_pred)

accuracy = (cm[0,0] + cm[1,1]) / (cm.sum())
error = (cm[1,0] + cm[0,1]) / (cm.sum())
precision = cm[0,0]/(cm[0,0] + cm[0,1])
recall = cm[0,0]/(cm[0,0] + cm[1,0])
print("Accuracy: {}, Error: {}, Precision: {}, Recall: {}".format(accuracy,
                                                                  error,
                                                                  precision, 
                                                                  recall))

