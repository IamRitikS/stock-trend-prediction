import re
import streamlit as st
from datetime import date
import datetime
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(
page_title = "Stock Trend Prediction",
page_icon = "📈"
)
st.title('Stock Trend Prediction')
hide_footer_style = """
<style>
#MainMenu { display: none; }
.viewerBadge_link__1S137 { display: none!important; }
footer {visibility: hidden!important;}
footer:after {content: "Made by Ritik Sharma"; visibility: visible; display: block; color: white; font-size:1em;}
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

stocks = ['AAPL - Apple Inc. Common Stock', 'ZS - Zscaler Inc. Common Stock', 'GOOGL - Alphabet Inc. Class A Common Stock', 'MSFT - Microsoft Corporation Common Stock']
with open("tickers.csv") as f:
    for row in f:
        ticker_search = re.search('^([^,]+),([^,]+),\$', row, re.IGNORECASE)
        if(ticker_search):
            ticker = ticker_search.group(1) + " - " + ticker_search.group(2)
            stocks.append(ticker)
#stocks = ('AAPL', 'MSFT')
selected_stock = st.selectbox('Enter stock ticker for prediction', stocks)
selected_stock = selected_stock.split(' ', 1)[0]

START = st.date_input(
     "Select start date of data",
     datetime.date(2015, 1, 1))
TODAY = date.today().strftime("%Y-%m-%d")

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

    
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('')

st.subheader('Raw data')
#st.write(data.tail())
st.write(data)

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
#st.write(forecast.tail())
st.write(forecast)
    
st.write(f'Forecast plot for {n_years} year(s)')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


dataToCompare = data[['Close']]
forecastToCompare = forecast.iloc[:dataToCompare.shape[0],:]
forecastToCompare = forecastToCompare[['yhat']]
st.success("Model was able to predict values with regression score of upto "+str(r2_score(dataToCompare, forecastToCompare)))
# print(r2_score(dataToCompare, forecastToCompare))