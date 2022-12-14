from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import streamlit as st

st.title("STOCK PREDICTION")
start_date = '2002-01-06'
end_date = datetime.today().strftime('%Y-%m-%d')

preset_stock = pd.read_csv('stock name.csv')
# get stock symbol from given company name
stock_name = st.selectbox('Select Stock', preset_stock['Name'])
stock_symbol = preset_stock[preset_stock['Name']
                            == stock_name]['Symbol'].values[0]

st.header("Stock Price")


def get_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start_date, end_date)
    return stock_data


status = st.text('Loading Stock Data')
stock_data = get_stock_data(stock_symbol, start_date, end_date)
stock_data.reset_index(inplace=True)
status.text(f'Stock Data Loaded: {stock_name}  Symbol:  {stock_symbol}')


st.subheader("Stock Price Data")
st.write(stock_data.tail())
# plot stock price
st.subheader("Stock Price plot")


def plot_stock_price(Stock_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Stock_data['Date'], y=Stock_data['Open'], name='Open'))
    fig.add_trace(go.Scatter(
        x=Stock_data['Date'], y=Stock_data['Close'], name='Close'))
    fig.add_trace(go.Scatter(
        x=Stock_data['Date'], y=Stock_data['High'], name='High'))
    fig.add_trace(go.Scatter(
        x=Stock_data['Date'], y=Stock_data['Low'], name='Low'))
    fig.update_layout(xaxis_title='Date', yaxis_title='Price',
                      xaxis_rangeslider_visible=True)
    fig.update_layout(
        title=stock_name,
        title_x=0.5,
        autosize=False,
        width=800,
        height=600,
        xaxis=dict(rangeselector=dict(
            buttons=list([
                dict(count=30, label="30D", step="day", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward")
            ])
        )))
    st.plotly_chart(fig)


plot_stock_price(stock_data)


def candle_plot(stock_data):

    fig = go.Figure(data=go.Candlestick(
        x=stock_data.Date,
        open=stock_data.Open,
        high=stock_data.High,
        low=stock_data.Low,
        close=stock_data.Close
    ))

    fig.update_layout(
        title=stock_name,
        title_x=0.5,
        autosize=False,
        width=800,
        height=600,
        xaxis=dict(rangeselector=dict(
            buttons=list([
                dict(count=1, label='1M', step="month", stepmode="backward"),
                dict(count=1, label='1Y', step="year", stepmode="backward")
            ])
        )),
    )
    st.plotly_chart(fig)


st.subheader("Stock Candlestick plot")
candle_plot(stock_data)

train = stock_data[['Date', 'Close']]
train = train.rename(columns={"Date": "ds", "Close": "y"})


st.header("Stock Prediction")

range = st.slider('Select the number of years to predict', 1, 40)
period = 365*range


@st.cache
def train_model(train, period):
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(period)
    forecast = model.predict(future)
    return model, forecast


prediction = st.text('Predicting Future of the Stock')
model, forecast = train_model(train, period)
prediction.text('Model Trained')
st.subheader('forecasted stock price')

def plot_prediction(model ,forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predicted'))
    fig.add_trace(go.Scatter(x= model.history['ds'], y=model.history['y'], name='Actual'))
    fig.update_layout(xaxis_title='Date', yaxis_title='Price',
                      xaxis_rangeslider_visible=True)
    fig.update_layout(
        title=stock_name,
        title_x=0.5,
        autosize=False,
        width=800,
        height=600,
        xaxis=dict(rangeselector=dict(
            buttons=list([
                dict(count=1, label='1M', step="month", stepmode="todate"),
                dict(count=1, label='1Y', step="year", stepmode="todate")
            ])
        )))
    st.plotly_chart(fig)

plot_prediction(model, forecast)


st.subheader('forcasted stock Components')
fig2 = model.plot_components(forecast)
st.write(fig2)

st.header('Stock Analysis')
last_index = len(forecast)-1
today_index = forecast.index[forecast['ds'] == end_date].tolist()[0]
profit = (forecast['yhat'][last_index]-forecast['yhat']
          [today_index])*100/forecast['yhat'][today_index]
pro = "{:.2f}".format(profit)
st.write('Profit: ', pro+'%')

if profit > 0:
    st.write('Prediction: Stock will rise')
else:
    st.write('Prediction: Stock will fall')

if profit < 6:
    st.write(f'Dont Even think Of touching this stock go buy an FD insted')

if profit > 6 and profit < 15:
    st.write(f'You can think of buying it for {range} years but you should be careful')

if profit > 15 and profit < 50:
    st.write(f'You can keep this for {range} years')

if profit > 50:
    st.write(f'You should have this stock in your porfoilio for {range} years')
