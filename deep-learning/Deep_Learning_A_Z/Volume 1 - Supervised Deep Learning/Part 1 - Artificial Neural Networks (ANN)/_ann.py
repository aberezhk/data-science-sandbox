# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:,-1:].values

# encode categorical variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

labelencoder_X1 = LabelEncoder()
labelencoder_X2 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1], dtype = np.float)
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# split data in test and train
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# drop-out reguralization - randomly disable some neurons at each iteration
def build_model(optimizer = 'adam'):
    # initialize ANN
    model = Sequential([
        Dense(6,kernel_initializer='uniform', activation="relu", input_shape=(11,)),
        Dropout(rate = 0.1),
        Dense(6, activation="relu"),
        Dense(1, activation="sigmoid"),
        ])
    # compile model
    model.compile(optimizer = optimizer, loss="binary_crossentropy", metrics=['accuracy'])
    return model

# simple example
model = build_model()
# fot model on train data
model.fit(X_train, Y_train, batch_size=10, epochs=100)
# predict on test data
y_pred = model.predict(X_test)
y_pred = (y_pred>0.5) # to get true/false

# evaluate model with confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

# homework - predict on single new data sample
new_val = scaler.transform(np.array([[
        0.0,0,600,1,40,3,10000,2,1,0,5000
        ]]))
new_prediction = model.predict(new_val)

# model evaluation with k-fold cross validation
# scikit learn api used as tensorwlof does not support cross validation out of box
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
model_kfold = KerasClassifier(build_fn=build_model, batch_size=10, epochs=100)
accuraces = cross_val_score(estimator=model_kfold, X=X_train, y=Y_train, cv=10)
mean_accuracy = accuraces.mean()
variance_accuracy = accuraces.std()

# parameter tuning
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
model_grid = KerasClassifier(build_fn=build_model)
# create dictionary of hyperparameters
parameters = {
        'batch_size': [25, 32],
        'epochs' : [100, 200],
        'optimizer' : ['adam', 'rmsprop']}
# define grid search object
grid_search = GridSearchCV(estimator=model_grid,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
