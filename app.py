import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

st.set_page_config(page_title="Ticker Stock Viz", page_icon="📈")
st.title("Ticker Stock Viz")

# --- Function Definitions ---

def render_sidebar():
    st.sidebar.title("Options")
    ticker = st.sidebar.text_input("Enter stock ticker:", "TSLA")
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))
    return ticker, start_date, end_date

@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

def display_indicators(data):
    st.header("Technical Indicators")
    # Calculate indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().abs()
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
    high_14, low_14 = data['High'].rolling(window=14).max(), data['Low'].rolling(window=14).min()
    k_percent = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
    d_percent = k_percent.rolling(window=3).mean()

    # Plotting
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)
    axes[0].plot(data['Close'], label='Close Price')
    axes[0].plot(data['MA50'], label='50-day MA')
    axes[0].legend()
    axes[0].set_ylabel('Price')
    axes[1].plot(rsi, label='RSI')
    axes[1].legend()
    axes[1].set_ylabel('RSI')
    axes[1].set_ylim(0, 100)
    axes[2].plot(k_percent, label='%K')
    axes[2].plot(d_percent, label='%D')
    axes[2].legend()
    axes[2].set_ylabel('Stochastic Oscillator')
    axes[2].set_ylim(0, 100)
    axes[3].plot(data['Volume'], label='Volume')
    axes[3].legend()
    axes[3].set_ylabel('Volume')
    st.pyplot(fig)

def plot_model_results(y_test, predictions, model_name, rmse, next_day_prediction):
    st.subheader(f"{model_name} Model")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test, label="Actual")
    ax.plot(predictions, label=f"{model_name} Predicted")
    ax.legend()
    ax.set_title(f"{model_name} Model")
    st.pyplot(fig)
    st.write(f'{model_name} Root Mean Squared Error:', rmse)
    st.write(f'{model_name} Next Day Prediction: {next_day_prediction}')

def display_ml_predictions(data):
    st.header("Machine Learning Models")
    # Data preprocessing
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    # Split data
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # --- LSTM ---
    def create_dataset(dataset, time_step=1):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    @st.cache_resource
    def train_lstm_model(X_train, y_train):
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50, return_sequences=False),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        return lstm_model

    time_step = 60
    X_train_lstm, y_train_lstm = create_dataset(train_data, time_step)
    X_test_lstm, y_test_lstm = create_dataset(test_data, time_step)
    X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
    X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

    lstm_model = train_lstm_model(X_train_lstm, y_train_lstm)
    lstm_predictions = lstm_model.predict(X_test_lstm)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    y_test_lstm_inv = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))
    lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm_inv, lstm_predictions))

    last_data_lstm = X_test_lstm[-1].reshape(1, time_step, 1)
    next_day_lstm_prediction = scaler.inverse_transform(lstm_model.predict(last_data_lstm))

    plot_model_results(y_test_lstm_inv, lstm_predictions, "LSTM", lstm_rmse, next_day_lstm_prediction[0][0])

    # --- SVM ---
    @st.cache_resource
    def train_svm_model(X_train, y_train):
        svm_model = SVR(kernel='linear')
        svm_model.fit(X_train, y_train.ravel())
        return svm_model

    X_train_svm, y_train_svm = train_data[:-1], train_data[1:]
    X_test_svm, y_test_svm = test_data[:-1], test_data[1:]

    svm_model = train_svm_model(X_train_svm, y_train_svm)
    svm_predictions = svm_model.predict(X_test_svm)
    svm_predictions = scaler.inverse_transform(svm_predictions.reshape(-1, 1))
    y_test_svm_inv = scaler.inverse_transform(y_test_svm.reshape(-1, 1))
    svm_rmse = np.sqrt(mean_squared_error(y_test_svm_inv, svm_predictions))

    last_data_svm = test_data[-1].reshape(1, -1)
    next_day_svm_prediction = scaler.inverse_transform(svm_model.predict(last_data_svm).reshape(1, -1))

    plot_model_results(y_test_svm_inv, svm_predictions, "SVM", svm_rmse, next_day_svm_prediction[0][0])

    # --- LightGBM ---
    @st.cache_resource
    def train_lgbm_model(X_train, y_train):
        lgbm_model = LGBMRegressor()
        lgbm_model.fit(X_train, y_train.ravel())
        return lgbm_model

    X_train_lgbm, y_train_lgbm = train_data[:-1], train_data[1:]
    X_test_lgbm, y_test_lgbm = test_data[:-1], test_data[1:]

    lgbm_model = train_lgbm_model(X_train_lgbm, y_train_lgbm)
    lgbm_predictions = lgbm_model.predict(X_test_lgbm)
    lgbm_predictions = scaler.inverse_transform(lgbm_predictions.reshape(-1, 1))
    y_test_lgbm_inv = scaler.inverse_transform(y_test_lgbm.reshape(-1, 1))
    lgbm_rmse = np.sqrt(mean_squared_error(y_test_lgbm_inv, lgbm_predictions))

    last_data_lgbm = test_data[-1].reshape(1, -1)
    next_day_lgbm_prediction = scaler.inverse_transform(lgbm_model.predict(last_data_lgbm).reshape(1, -1))

    plot_model_results(y_test_lgbm_inv, lgbm_predictions, "LightGBM", lgbm_rmse, next_day_lgbm_prediction[0][0])

# --- Main Application Logic ---

def main():
    ticker, start_date, end_date = render_sidebar()

    tabs = ["Indicators", "Machine Learning"]
    selected_tab = st.sidebar.selectbox("Select tab:", tabs)

    try:
        data = load_data(ticker, start_date, end_date)
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
            return

        if selected_tab == "Indicators":
            display_indicators(data)
        elif selected_tab == "Machine Learning":
            display_ml_predictions(data)

    except ValueError as e:
        st.error(f"Error fetching data for ticker {ticker}: {e}")
    except Exception as e:
        st.error("An unexpected error occurred. Please try again later.")
        # In a real application, you would log the full error e here
        # logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
