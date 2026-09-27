# Stock Trend Prediction

[![Streamlit](https://img.shields.io/badge/Streamlit-1.11.0-orange.svg)](https://streamlit.io/)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Prophet](https://img.shields.io/badge/Prophet-Time%2520Series-yellow.svg)](https://facebook.github.io/prophet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive web application built with **Streamlit** that fetches historical stock market data and leverages **Facebook Prophet** for time-series forecasting and trend analysis.

---

## 🚀 Key Features

* **Interactive Web Interface:** Clean dashboard powered by Streamlit for seamless user interaction.
* **Dynamic Ticker Selection:** Choose from default major stocks (AAPL, ZS, GOOGL, MSFT) or parse custom symbols dynamically from `tickers.csv`.
* **Customizable Date & Range:** Select custom historical start dates and adjust forecast horizons from 1 to 4 years using interactive sliders.
* **Interactive Visualizations:** View raw data time-series charts with range sliders, forecast trajectories, and individual trend components (seasonality, weekly trends) using **Plotly**.
* **Model Evaluation:** Automatically calculates and displays the regression ($R^2$ score) performance metric comparing actual historical closes against model predictions.

---

## 🛠️ Tech Stack

* **Framework:** Streamlit
* **Time-Series Forecasting:** Facebook Prophet (`fbprophet`)
* **Financial Data Source:** Yahoo Finance (`yfinance`)
* **Data Visualization:** Plotly (`plotly.graph_objs`)
* **Evaluation Metrics:** Scikit-Learn (`scikit-learn`)
* **Data Manipulation:** Pandas, NumPy

---

## 📂 Project Structure

```text
stock-trend-prediction/
│
├── main.py              # Core Streamlit application script
├── tickers.csv          # CSV database of stock symbols, names, and exchange details
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation

```

---

## ⚙️ Installation & Setup

Follow these steps to run the application locally on your machine:

1. **Clone the repository:**
```bash
git clone [https://github.com/IamRitikS/stock-trend-prediction.git](https://github.com/IamRitikS/stock-trend-prediction.git)
cd stock-trend-prediction

```


2. **Create and activate a virtual environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate     # On Windows use: venv\Scripts\activate

```


3. **Install the required dependencies:**
```bash
pip install -r requirements.txt

```



---

## 📊 Usage Guide

Run the Streamlit application from your terminal:

```bash
streamlit run main.py

```

Once running, the local web server will open in your browser allowing you to:

1. Select or search for a stock ticker from the dropdown menu.
2. Choose your historical start date.
3. Slide to pick the number of years for prediction (1–4 years).
4. Analyze the raw data table, interactive Plotly charts, Prophet forecast curves, component breakdowns, and model $R^2$ accuracy score.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or pull request.

---

## 📝 License

Distributed under the **MIT License**. See `LICENSE` for more information.
