from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def regression_nn():

    y_train = pd.read_csv('y_train.csv')
    X_train = pd.read_csv('X_train.csv')
    num_in = X_train.shape[-1] # get the number of features
    regressor = Sequential()
    regressor.add(Dense(output_dim=int(num_in + 1/2), init='uniform',
                         activation='relu', input_dim=num_in))

    regressor.add(Dense(output_dim=20, init='uniform',
                        activation='relu'))

    regressor.add(Dense(output_dim=8, init='uniform',
                        activation='relu'))

    regressor.add(Dense(output_dim=1, init='uniform',
                        activation='linear'))

    regressor.compile(optimizer='adam', loss='mean_squared_error',
                       metrics=['accuracy'])

    regressor.fit(X_train, y_train, batch_size=50, epochs=1000, shuffle=True)

    X_test = pd.read_csv('X_test.csv')
    y_pred = regressor.predict(X_test)

    y_pred = pd.DataFrame(data=y_pred)
    y_pred.to_csv('y_pred.csv', index=False)

# pd.concat([survey_sub, survey_sub_last10], axis=1)

if __name__ == '__main__':
    regression_nn()

