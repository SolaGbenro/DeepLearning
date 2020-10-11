"""
    Sola Gbenro:
        Deep Learning A-Z -- Part 3 Recurrent Neural Network
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# save flag
SAVE = True
# import training data
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# grab all rows, but only the 'Open' column values (we will be predicting opening price for Jan 2017)
training_set = dataset_train.iloc[:, 1:2].values    # [:, 1] creates a single vector, we need numpy array

# Feature Scaling, for RNN Normalisation is recommended over Standardisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
# apply feature scalar
training_set_scaled = sc.fit_transform(training_set)

"""
    "Creating a data structure with 60 timesteps and 1 output"

        View previous 60 values in calculating next value. For each observation, X_train will contain the previous 60
        'Open' prices and y_train will contain the next financial days 'Open' price.
"""
X_train = []
y_train = []
# last index in observations = 1258
for i in range(60, 1258):
    # take a rolling 60 observations
    X_train.append(training_set_scaled[i-60:i, 0])
    # upperbound is excluded, take i instead of i+1
    y_train.append(training_set_scaled[i, 0])

# convert to numpy arrays for RNN input
X_train, y_train = np.array(X_train), np.array(y_train)

"""
    Add dimension to training data, required by RNN (also adds ability to add additional features such as close price)
        From Keras: 3D tensor with shape (batch_size, timesteps, input_dim)
            - batch_size here is equal to total observations (1258)
            - timesteps is equal to count of previous observations included in calculation (60)
            - input_dim is number of indicators/predictors or features included as independent variable
"""
# X_train.shape[0] = len of rows
# X_train.shape[1] = num of cols
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# create model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
# Add first layer
# return_sequences = True for ALL but the last layer of LSTM (want to feedforward)
# input_shape = num timesteps and number of indicators/predictors/features
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Add second layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Add third layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Add fourth layer, this will be last LSTM layer (no return_sequences)
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# Add final/output layer, output will have single dimension
regressor.add(Dense(units=1))

# compile the model
regressor.compile(optimizer='adam', loss='mean_squared_error')

# fit model on training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# save model
if SAVE is True:
    regressor.save(os.getcwd() + '\models')

# load test_set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# view prediction and test
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
