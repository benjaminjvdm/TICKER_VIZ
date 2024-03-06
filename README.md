**Title:** Tesla Stock Price Analysis and Prediction Web App

**Brief Description**

This project provides a user-friendly web application built with Streamlit to analyze Tesla (TSLA) stock prices and make predictions using various machine learning models. The app offers the following features:

* **Historical Stock Price Visualization:**  View Tesla's closing prices along with technical indicators (RSI, Stochastic Oscillator).
* **Date Filtering:** Customize the displayed date range using an interactive sidebar.
* **Prediction Models:** Explore predictions generated by models including:
   * LSTM (Long Short-Term Memory)
   * SVM (Support Vector Machine)
   * LightGBM 

**Data Source**
Stock data is retrieved from Yahoo Finance (yfinance)

**How to Run**

1. **Install Dependencies:**
   ```bash
   pip install requirements.txt
   ```

2. **Download the Code:** Clone or download this repository. 

3. **Run the App:**
    ```bash
    streamlit run app.py  # Replace 'app.py' if your file has a different name
    ```
   The app will launch in your web browser, typically at http://localhost:8501

**Code Structure**

* **app.py (or similar):** Contains the main Streamlit application logic.
* **Import Statements:**  Ensure the code starts with necessary imports.
* **Functions:** Each core functionality is organized into well-defined functions (e.g., `visualize_stock_price_history`, `build_and_train_model`, etc.)

**Using the Web App**

1. **Sidebar Options:** Modify the date range and choose which indicators to display in the stock price chart.
2. **Analysis Sections:** Select the type of analysis you wish to perform:
   *  "Stock Price History" for pure visualization with indicators.  
   *  "All Models" to compare prediction results from different models
   *  Specific model options (LSTM, SVM, LightGBM) to focus on a single model.

**Potential Improvements**

* **Additional Indicators:** Add more technical indicators depending on your analysis needs.
* **Model Refinement:** Experiment with different model architectures and hyperparameters.
* **Error Handling:** Implement error handling for cases of insufficient data.
* **Deployment/Hosting:** Explore cloud-based services to make the app accessible to a wider audience.

**Disclaimer**
This code and application are intended for educational and exploratory purposes. They should not be considered a substitute for professional financial advice.
