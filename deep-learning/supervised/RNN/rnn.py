# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# scale features by normalization: (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# create data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60, training_set_scaled.shape[0]):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# reshape to match lstm input from ts.keras (needs 3d tensor)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Importing the ts libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Initialising the RNN as a sequential model
# it will have 4 lstm layers and 1 output layer
regressor = Sequential()

# 1st layer - lstm with 50 neurons
# units - number of neurons at given layer
# return_sequences - stack layers is set to true because we will chain other lst layers on top
# shape - number of timestamps (60) by number of predictors (1)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
# add dropout to prevent overfitting - 10 neurons will be deactivated randomly
regressor.add(Dropout(0.2))

# 2nd layer - no need to specify shape as it is not a first layer 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# 3rd layer - same as 2nd
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# 4th layer - stop returning sequences as it is last lstm layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# output layer - Dense - has 1 output values - thus units = 1
regressor.add(Dense(units=1))

# compile the model
# adam optimiziation function (though rmsprop is usual for lstm)
# loss mean_squared_error - as it is a regression analyses
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# for model to the data
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

# get test data (january 2017 google stock price)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:, 1:2].values

# make predictions
# to predict each day we need 60 previous prices for each day - thus we will need both training and test sets
# concatinate train set and test set (not scaled), take only opening price and join on rows (axis=0)
# ! model was trained on scaled data but we can not change test data
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

# get the inputs - 60 previous dates for each new date
# take last day of training data and subtract 60 (as we use 60 timestamps for prediction)
first_to_use = len(dataset_total) - len(dataset_test) - 60 
inputs = dataset_total[first_to_use:].values # from training - 60 to the last timestamp to predict the last day of january
# reshape values from vector (80,) to np array (80,1) 
inputs = inputs.reshape(-1,1)
# scale inputs using the same scaler we used for training (do not fit data again, only transform)
inputs = sc.transform(inputs)

# prepare same data structure as for training (1 line has 60 previous timestamps)
x_test = []
for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
# to reverse normalisation and get real values again
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(test_set, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



