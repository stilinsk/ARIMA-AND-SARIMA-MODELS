# ARIMA-AND-SARIMA-MODELS

### PROJECT OVERVIEW
In the following model we are implementing a time series analysis based on two commonly used models (ARIMA  and SARIMA MODELS) Where each is used in different sitiuations based on the different types of data ,However we will use both models in our data and expalin why we are using a specific type of  model based on different types and forms of data.
ARIMA (AutoRegressive Integrated Moving Average) and SARIMA (Seasonal AutoRegressive Integrated Moving Average) models are time series forecasting techniques used to analyze and predict time-dependent data. These models are valuable tools in various fields, including economics, finance, epidemiology, and environmental science. Here's an overview of both ARIMA and SARIMA models:

**ARIMA Models (AutoRegressive Integrated Moving Average):**

1. **AutoRegressive (AR) Component**: The ARIMA model includes an autoregressive component, denoted as "AR(p)," where "p" represents the order of the autoregressive component. It captures the relationship between the current value of the time series and its past values.

2. **Integrated (I) Component**: The "I" stands for integrated. It represents the number of differences needed to make the time series stationary, i.e., constant mean and variance. This differencing step helps remove trends and seasonality from the data.

3. **Moving Average (MA) Component**: The ARIMA model also includes a moving average component, denoted as "MA(q)," where "q" represents the order of the moving average component. It models the relationship between the current value and past forecast errors.

The ARIMA model is often denoted as ARIMA(p, d, q), where "p" is the order of the autoregressive component, "d" is the degree of differencing, and "q" is the order of the moving average component.

**SARIMA Models (Seasonal AutoRegressive Integrated Moving Average):**

SARIMA models extend ARIMA models by incorporating seasonality into the model. They are suitable for time series data that exhibit seasonality patterns, such as daily, monthly, or yearly fluctuations.

In addition to the ARIMA components (AR, I, MA), SARIMA models include:

4. **Seasonal AutoRegressive (SAR) Component**: This component, denoted as "SAR(P, D, Q, s)," captures the seasonal patterns in the data. "P" represents the seasonal autoregressive order, "D" is the seasonal differencing degree, "Q" is the seasonal moving average order, and "s" is the season's length.

SARIMA models are denoted as SARIMA(p, d, q)(P, D, Q, s), where the first set of parameters represents the non-seasonal components, and the second set represents the seasonal components.

Steps to Build ARIMA and SARIMA Models:

1. **Data Preparation**: Clean and preprocess the time series data, ensuring it's stationary if using ARIMA or SARIMA models.

2. **Model Identification**: Identify the appropriate values of "p," "d," "q," "P," "D," "Q," and "s" using techniques like autocorrelation and partial autocorrelation plots.

3. **Model Estimation**: Estimate the model parameters based on the identified orders.

4. **Model Diagnosis**: Check the residuals for stationarity and white noise properties. Refine the model if needed.

5. **Model Forecasting**: Use the fitted ARIMA or SARIMA model to make future predictions.

6. **Model Evaluation**: Evaluate the model's performance using appropriate metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE).

Both ARIMA and SARIMA models provide a powerful framework for time series forecasting when dealing with data that exhibits trends, seasonality, and autocorrelation patterns. However, selecting the right model orders and properly diagnosing the models are critical steps in achieving accurate forecasts.
  ##We will start the implementation of out model where we will be using data from 2015 to 2020

  we will  be using teh following libraries for data eda
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
### Data cleaning and preparation
In the initial preprocessing of our data we will

1.Load our data

2.Check for missing values

3.Handle the missing values  -However this process is determined by theamount of data that is missing (when the data is fairly small then we drop the columns and when there is a large amount of data missing we need to fill the missing values ith the median as this ensures that our data is not skewed towards a particular direction.

4.Data cleaning and formatting .Since its a tie series data we have to ensure that our date column is in a datetiem format and that we only have the  two columns that we are using for prediction its alays advisable to drop columns where we are not using for prediction,thus we are only left with two columns.

## Explolatory data analysis
Eda is exploring the sales data to answer key questions such as 
- which years do we have more sales and lesser sales
- what is the trend of our data is our data stationery or seasonal ( The type of data dicated which model we will be using SARIMA for seasonal data  and ARIMA  for stationery data
-The monthly trend of our data (since our data is daily we will have to later group the data  to monthly
- we will also be checkingt he roling mean and standard deviation this will help  look at the trend of our data instead of guessing the trend ,improves our accuracy

  ## findings and conclusions
  
  -The data from all the trends and the plots we can see our data is seasonal
  
  -When the year is midway we tend ti have lower sales
  
  -When the year is approcahing towards the end we have higher amounrt of sales recorded

  ## Model building
  Since we have already come to the conclusion that our data is seasonal we will have now to find ways of finding our parameters for fitting our model for good perfomance but note we will be using bothe models ARIMA and SARIMA to understand why its advisable to use each model for different type of data

  We will start by figuring out our d value. The value od d is either 0 or 1 if the data is stationery we should use 0 and if the data is seasonal we should use 1

  In the below line of code we will be looking for the value of p the value of p we would simply put it as after which line do we see our plotted line distend and have the biggest change in the curve . we look at that line and after we see it then we can actually determine the value of p
  
  ` pd.plotting.autocorrelation_plot(df["Sales_quantity"]) `

  In the below code we will be looking for the value of q(moving average).in the below plots we find the correlation and that why the first figure plotted will always coincide with 1and thus to find q we look at where from the first plot do we start risig againand its quite clear we start rising from the third plot and thus our q is eqal to 3

```
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df["Sales_quantity"], lags = 20)
```
As we saw earlier that our  data is seasonal but lets first implement the ARIMA model
### Implementing the ARIMA MODEL
```
rom statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')



# Define the p, d, and q parameters to take any value between 0 and 2
p, d, q = 2, 1, 2

# Fit the model
model = ARIMA(df["Sales_quantity"], order=(p,d,q))
fitted = model.fit()

# Make predictions
predictions = fitted.predict(start=len(df), end=len(df)+100)

print(predictions)
```
Then we will plot it
```
df["Sales_quantity"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")
```

![ARIMA](https://github.com/stilinsk/ARIMA-AND-SARIMA-MODELS/assets/113185012/90699bce-8da2-4d4b-8823-7310c0c818fb)

Its quite clear that our model is perfoming below expectation and thus using Arima model it would be very unwise
### IMPLEMENTING THE SARIMA MODEL

In the code below we will be now using the sarima model and we willbe qrouping the values and we are using PDQ and fixing the values to 1 to assume that for the whole year we have one pq and d and it repeats itself every 12 months
```
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')


# Define the p, d, and q parameters to take any value between 0 and 2
p, d, q = 2, 1, 2

# Define the seasonal p, d, and q parameters to take any value between 0 and 2
P, D, Q = 1, 1, 1
m = 12 # Monthly data

# Fit the model
model = SARIMAX(df['Sales_quantity'], order=(p, d, q), seasonal_order=(P, D, Q, m))
fitted = model.fit()

# Make predictions for the next 12 months
start_date = pd.to_datetime(df.index.max()) + pd.DateOffset(months=1)
end_date = start_date + pd.DateOffset(months=11)
predictions = fitted.predict(start=start_date, end=end_date)

print(predictions)
````
We will plot how the Sarima model will perfom prediction on data for the next year and we will see how different it is from how the ARIMA MODEL  perfoms
```
# Make predictions for the next 12 months
predictions = fitted.predict(start=len(df), end=len(df)+11, dynamic=True)

# Plot the actual and predicted sales
plt.figure(figsize=(10,5))
plt.plot(df.index, df['Sales_quantity'], label='Actual')
plt.plot(predictions.index, predictions, label='Predicted')
plt.title('Actual vs Predicted Sales Quantity')
plt.xlabel('Date')
plt.ylabel('Sales Quantity')
plt.legend()
plt.show()
```

![AARIMA](https://github.com/stilinsk/ARIMA-AND-SARIMA-MODELS/assets/113185012/a40da2e6-3cc1-42d7-81d1-9a5568aaf4a8)

Thats is basically the explained indepth of ARIMA and SARIMA  MODELS

