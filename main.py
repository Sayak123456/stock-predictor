import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import pandas_datareader as data
import datetime as datetime
from keras.models import load_model
import tensorflow as tf

start=datetime.date.today()-datetime.timedelta(days=5*365)
end=datetime.date.today()-datetime.timedelta(days=1)

st.title("Stock Price Prediction")
input=st.text_input("Enter Stock Name [in CAPITALS]","AAPL")
try:
    df=data.DataReader(input,'yahoo',start,end)
    st.subheader("Last 10 days data")
    st.write(df.tail(10))
    st.subheader("Data Description")
    st.write(df.describe())
    st.subheader("Actual Closing Price Trend")
    fig=plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    X_train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    X_test=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    sc=MinMaxScaler(feature_range=(0,1))

    X_train_array=sc.fit_transform(X_train)

    X=[]
    y=[]
    for i in range(100,X_train_array.shape[0]):
        X.append(X_train_array[i-100:i])
        y.append(X_train_array[i,0])
    X,y=np.array(X),np.array(y)

    keras_model = load_model("LSTM_model (2).h5")
    #convert=tf.lite.TFLiteConverter.from_keras_model(keras_model)

    last_100days=X_train.tail(100)
    final_df=last_100days.append(X_test,ignore_index=True)
    input_data=sc.fit_transform(final_df)
    x_test=[]
    y_test=[]
    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
    x_test,y_test=np.array(x_test),np.array(y_test)

    y_predict=keras_model.predict(x_test)

    scale_factor=1/sc.scale_
    y_predict=y_predict*scale_factor
    y_test=y_test*scale_factor

    st.subheader("Original trend vs Predicted Trend (5 years)")
    fig1=plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original Data')
    plt.plot(y_predict,'r',label='Predicted Data')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig1)

except:
    st.subheader("Sorry! No data found")
