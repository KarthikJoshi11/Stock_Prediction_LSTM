# Stock Price Prediction using LSTM

This repository contains a Python script that uses a Long Short-Term Memory (LSTM) neural network to predict future stock prices based on historical data. The model is built using Keras and TensorFlow.

## Key Features

- **Time-Series Forecasting:** The project uses a multi-step time-series approach with a `look_back` window to capture temporal dependencies in the data.
- **Data Preprocessing:** It includes a custom preprocessing script (`preprocessing.py`) to prepare the data for the LSTM model, including min-max scaling and sequence creation.
- **Model Architecture:** The model consists of two LSTM layers and a final dense layer for regression, optimized with the Adam optimizer.
- **Visualization:** The script generates plots to visualize the original data, the model's training predictions, and test predictions to easily assess performance.
- **Performance Evaluation:** Model performance is evaluated using the Root Mean Squared Error (RMSE) on both the training and test sets.

## Files in this Repository

- `StockPrediction.py`: The main script that loads the data, builds and trains the LSTM model, and visualizes the results.
- `preprocessing.py`: A utility script containing the `new_dataset` function, which is essential for creating the time-series sequences for the LSTM.
- `apple_share_price.csv`: The dataset used for training and testing the model.

## How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  **Install the required libraries:**
    It is highly recommended to use a virtual environment to avoid dependency conflicts.
    ```bash
    # Create a virtual environment
    python3 -m venv venv

    # Activate the virtual environment
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install the necessary packages
    pip install pandas numpy scikit-learn matplotlib keras tensorflow
    ```

3.  **Run the script:**
    ```bash
    python StockPrediction.py
    ```
    This will execute the code, train the model, and display the performance metrics and a plot of the results.
