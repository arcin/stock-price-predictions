import pandas as pd
import pandas_datareader.data as web
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# Grab tesla stock from 9/9/2018 to now.
start = datetime.datetime(2018, 9, 9)
end = datetime.datetime.now()
df = web.DataReader("TSLA", "yahoo", start, end)

"""
Generate Features
- High Low Percentage (https://school.stockcharts.com/doku.php?id=index_symbols:high_low_percent)
- Percentage Change (https://www.mathsisfun.com/numbers/percentage-change.html)
"""
# Grab adjusted close and volume from data frame
regression_dataframe = df.loc[:, ['Adj Close', 'Volume']]
# Generate the featuers

regression_dataframe['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100
regression_dataframe['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100


"""
Pre-process data and perform cross validation
"""
# Replace any NaN values with -99999
regression_dataframe.fillna(value=-99999, inplace=True)

# 1% of data is separated for forecast. This will tell us how much 1% of the dataset is.
forecast_out = int(math.ceil(0.01 * len(regression_dataframe)))

# Separate our label (dependant variable) to prep for prediction. But remove forecast amount
# to leave a gap for prediction.
forecast_col = 'Adj Close'
regression_dataframe['label'] = regression_dataframe[forecast_col].shift(-forecast_out)
X = np.array(regression_dataframe.drop(['label'], axis=1)) # Drop our label from the data set and create a NumPy 2D array.

# Scale the X so that everyone can have the same distribution for linear regression

X = preprocessing.scale(X)

# Finally we want to find the Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(regression_dataframe['label'])
y = y[:-forecast_out]

"""
Split data for training and testing.
"""
# Separate training data into a training set and a test set,
# 80% for training, 20% for test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""
Train our linear models
- Linear Regression
- Ridge Regression
- Lasso Regressio
"""
# Linear regression
linear_regression = LinearRegression(n_jobs=-1)
linear_regression.fit(X_train, y_train)

# Ridge Regression
ridge_regression = Ridge(alpha=0.2)
ridge_regression.fit(X_train, y_train)

# Lasso Regression
lasso_regression = Lasso(alpha=0.2)
lasso_regression.fit(X_train, y_train)



#
#   Simple helper function that will plot the predictions for a linear model.
#
def plotLinearModel(plotName, model, dataset, X_forecast, start_date):
    forecast_set = model.predict(X_forecast)
    dataset['Forecast'] = np.nan

    last_unix = start_date
    next_unix = last_unix + datetime.timedelta(days=1)

    # Map price predictions to date of prediction.
    for i in forecast_set:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        price = [np.nan for _ in range(len(dataset.columns)-1)]+[i]
        dataset.loc[next_date] = price

    # Plot predictions
    dataset['Adj Close'].tail(500).plot()
    dataset['Forecast'].tail(500).plot()
    plt.legend(loc=4)
    plt.title(plotName)
    plt.xlabel('Date')
    plt.ylabel('Price')


"""
Plot our predictions
"""

forecast_start = regression_dataframe.iloc[-1].name
plt.figure(figsize=(10,10))

# Plot all our models as subplots.
plt.subplot(2, 2, 1, aspect='equal')
print('Confidence for Linear regression is', linear_regression.score(X_test, y_test))
plotLinearModel('Linear Regression', linear_regression, regression_dataframe, X_lately, forecast_start)

plt.subplot(2, 2, 2, aspect='equal')
print('Confidence for Ridge regression is', ridge_regression.score(X_test, y_test))
plotLinearModel('Ridge Regression', ridge_regression, regression_dataframe, X_lately, forecast_start)

plt.subplot(2, 2, 3, aspect='equal')
print('Confidence for Lasso regression is', lasso_regression.score(X_test, y_test))
plotLinearModel('Lasso Regression', lasso_regression, regression_dataframe, X_lately, forecast_start)
plt.show()

