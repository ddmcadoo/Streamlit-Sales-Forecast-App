#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import xgboost as xgb

st.title('Sales Data Forecasting')

# File uploader
file = st.file_uploader("Upload CSV file", type="csv")
if file is not None:
    df = pd.read_csv(file)
    st.write(df.head())  # Show the first few rows of the uploaded file

    # Assuming the file has 'date' and 'sales' columns
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week

    X = df[['day','month','week']]
    y = df['sales']

    model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=5)
    model.fit(X, y)

    future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=30)
    future_df = pd.DataFrame({'date': future_dates})
    future_df['day'] = future_df['date'].dt.dayofyear
    future_df['month'] = future_df['date'].dt.month
    future_df['week'] = future_df['date'].dt.isocalendar().week

    future_forecast = model.predict(future_df[['day', 'month', 'week']])


    forecast_df = pd.DataFrame({
        'day': future_df['day'],
        'predicted_sales': future_forecast
    })

    st.write("Forecast for the next 30 days:")
    st.write(forecast_df)
    
    # Allow users to download the forecast as a CSV
    st.download_button(
        label="Download Forecast",
        data=forecast_df.to_csv(index=False),
        file_name="forecast.csv",
        mime="text/csv"
    )


# In[ ]:




