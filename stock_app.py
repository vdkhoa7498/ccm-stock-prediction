import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from IPython.core.debugger import set_trace
from datetime import datetime,date,timedelta

import os
import time
from xgboost import XGBRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

XGBmodel = XGBRegressor()
XGBmodel.load_model("xgboost_close_model.txt")

app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

df_nse = pd.read_csv("./btc.csv")
df_nse = df_nse.sort_values(by='Date')
df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']

data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

percent_train = 80
percent_valid = 100 - percent_train
cout_percent_train = int(len(new_data)/100*percent_train)
cout_percent_valid = len(new_data)-cout_percent_train


df_for_xgboost = data
df_for_xgboost.head(5)
df_for_xgboost = df_for_xgboost[["Close"]].copy()
df_for_xgboost["Target"] = df_for_xgboost.Close.shift(-1)
df_for_xgboost.dropna(inplace=True)
df_for_xgboost.head(5)

def train_test_split(data, percent):
    data = data.values
    n = int(len(data) * (1 - percent))
    return data[:n], data[n:]

train, test = train_test_split(df_for_xgboost, 0.1)

predictions = []
for i in range(len(test)):
    inputVal = test[i, 0]

    val = np.array(inputVal).reshape(1, -1)
    pre = XGBmodel.predict(val)
    predictions.append(pre[0])

def train_test_split_plotdata(data, percent):
    n = int(len(data) * (1 - percent))
    return data[n:]

validXGB = train_test_split_plotdata(new_data,0.1)
validXGB["Predictions"] = predictions

dataset=new_data.values

train=dataset[0:cout_percent_train,:]
valid=dataset[cout_percent_train:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model_RNN=load_model("./model_RNN.h5")
model_LSTM=load_model("./model_LSTM.h5")


inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
RNN_closing_price=model_RNN.predict(X_test)
RNN_closing_price=scaler.inverse_transform(RNN_closing_price)
LSTM_closing_price=model_LSTM.predict(X_test)
LSTM_closing_price=scaler.inverse_transform(LSTM_closing_price)
XGB_closing_price=model_LSTM.predict(X_test)
XGB_closing_price=scaler.inverse_transform(XGB_closing_price)

train=new_data[:cout_percent_train]
validRNN=new_data[cout_percent_train:]
validRNN['Predictions']=RNN_closing_price
validLSTM=new_data[cout_percent_train:]
validLSTM['Predictions']=LSTM_closing_price

validXGB1=new_data[cout_percent_train:]
validXGB1['Predictions']=XGB_closing_price



df= pd.read_csv("./stock_data.csv")

app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Predict Stock Close Price',children=[
            html.Div([
                html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=validRNN.index,
                                y=validRNN["Close"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                ),
                html.H2("XGBoost Predicting Close price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=validRNN.index,
                                y=validXGB1["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Predict Closing Rate'}
                        )
                    }

                ),
                html.H2("RNN Predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data RNN",
                    figure={
                        "data":[
                            go.Scatter(
                                x=validRNN.index,
                                y=validRNN["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                ),
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data LSTM",
                    figure={
                        "data":[
                            go.Scatter(
                                x=validLSTM.index,
                                y=validLSTM["Predictions"],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                ),                
            ])                


        ])
    ])
])


if __name__=='__main__':
    app.run_server(debug=True)