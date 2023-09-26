# ARIMA-AND-SARIMA-MODELS
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
  # We will start the implementation of out model where we will be using data from 2015 to 2020
