🌦️ Weather Forecasting Prediction System

A Machine Learning-based project that predicts daily average temperature using historical weather data and a Random Forest Regression model.

📌 Project Overview

Weather forecasting plays an important role in daily life, agriculture, transportation, and planning outdoor activities. This project uses historical weather data to build a Machine Learning model that estimates temperature for upcoming dates.

The system takes a date as input and generates temperature predictions for four consecutive days. It also displays a simple temperature-based weather label.

Project Type: Machine Learning | Regression
Programming Language: Python
Model: Random Forest Regressor
Platform: Google Colab

🎯 Objectives

* Analyze historical weather data.
* Perform data preprocessing and feature engineering.
* Train a Random Forest Regression model.
* Predict daily average temperature for upcoming dates.
* Evaluate model performance using regression metrics.
* Generate four-day temperature estimates based on user input.

✨ Features

* Historical weather data analysis
* Date and time feature extraction
* Missing-value handling
* Random Forest Regression
* Four-day temperature prediction
* Model evaluation using MAE and R² score
* Trained model saving using Joblib
* Temperature-based weather labels

📂 Dataset Description

The project uses historical daily weather data for India covering the period 2000–2024.

The dataset contains weather measurements such as:

* city — city associated with the weather record
* date — date of the weather observation
* temperature_2m_max — maximum daily temperature at 2 metres
* temperature_2m_min — minimum daily temperature at 2 metres
* apparent_temperature_max — maximum apparent temperature
* apparent_temperature_min — minimum apparent temperature
* precipitation_sum — total precipitation
* rain_sum — total rainfall
* weather_code — numerical weather condition code
* wind_speed_10m_max — maximum wind speed at 10 metres
* wind_gusts_10m_max — maximum wind gusts
* wind_direction_10m_dominant — dominant wind direction

The exact target variable and input features depend on the training code used in the notebook.

🛠️ Technologies Used

Technology	Purpose
Python	Programming language
Pandas	Data manipulation and preprocessing
NumPy	Numerical computations
Scikit-learn	Machine Learning and model evaluation
Random Forest Regressor	Temperature prediction
Joblib	Saving and loading trained models
Google Colab	Development and execution environment

⚙️ Methodology

1. Data Collection: Load the historical weather dataset.
2. Data Analysis: Inspect the dataset structure, columns, and missing values.
3. Data Preprocessing: Clean the data and convert date values into datetime format.
4. Feature Engineering: Extract year, month, and day from the date.
5. Feature Selection: Select date-based input features and the temperature target.
6. Model Training: Train the Random Forest Regression model.
7. Model Evaluation: Evaluate performance using MAE and R² score.
8. Model Saving: Save the trained model using Joblib.
9. Prediction: Accept a user-input date and estimate temperature for four consecutive days.

🤖 Machine Learning Algorithm

Random Forest Regression

Random Forest Regression is an ensemble Machine Learning algorithm that combines predictions from multiple decision trees to estimate a continuous numerical value.

In this project, the algorithm is used to predict daily average temperature from date-based features.

Advantages:

* Can model nonlinear relationships.
* Combines multiple decision trees.
* Works with numerical input features.
* Provides a useful baseline for regression tasks.

📊 Model Evaluation and Output Measures

The initial model produced the following evaluation results on the test split:

Metric	Result
Mean Absolute Error (MAE)	2.5255
R² Score	0.5305
Algorithm	Random Forest Regressor
Prediction Target	Temperature

1. Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

Formula:

MAE = (1/n) × Σ |Actual − Predicted|

Observed result: 2.5255

If the target variable is measured in degrees Celsius, the MAE is approximately 2.53°C. Lower MAE values indicate smaller average prediction errors.

2. R² Score

R² measures how much of the variation in the target values is explained by the model relative to a mean-value baseline.

Formula:

R² = 1 − (SS_res / SS_tot)

Observed result: 0.5305

The model explains approximately 53.05% of the variation in the test target values under the reported evaluation setup. This is not the same as 53.05% prediction accuracy.

Performance Interpretation

The initial model provides a baseline for temperature estimation. Its performance can potentially be improved through additional weather features, time-series lag features, hyperparameter tuning, and chronological validation.

Evaluation note: The reported metrics came from a random train-test split. Since weather data is time-dependent, a chronological train-test split is recommended to evaluate future forecasting performance more realistically.

🖥️ Sample Input and Output

Sample Input

Enter Year: 2025
Enter Month: 7
Enter Day: 10

Sample Output Format

========== WEATHER FORECAST ==========
Forecast for Next 4 Days
Date: 10-07-2025
Predicted Temperature: XX.XX °C
Weather: Temperature-based label
Date: 11-07-2025
Predicted Temperature: XX.XX °C
Weather: Temperature-based label
Date: 12-07-2025
Predicted Temperature: XX.XX °C
Weather: Temperature-based label
Date: 13-07-2025
Predicted Temperature: XX.XX °C
Weather: Temperature-based label
Prediction Completed

Note: XX.XX is a placeholder. Actual temperature values are generated by the trained model when the code runs.

The current basic implementation assigns weather labels using temperature thresholds. These labels do not represent independently predicted rainfall, cloud cover, or official weather conditions.

🚀 How to Run the Project

Requirements

* Google account
* Google Colab
* Historical weather CSV dataset
* Python libraries

Installation

Install the required libraries in Google Colab:

pip install pandas numpy scikit-learn joblib

Execution Steps

1. Open Google Colab.
2. Upload the project notebook.
3. Upload the required weather CSV dataset.
4. Run the notebook cells in order.
5. Train the Random Forest Regression model.
6. Review the MAE and R² evaluation results.
7. Enter a date when prompted.
8. View the four-day temperature estimates.

📁 Project Structure

weather-prediction/
│
├── project_1.ipynb
├── weather_forecast.py
├── requirements.txt
└── README.md

* project_1.ipynb — notebook containing data analysis, model training, and experiments.
* weather_forecast.py — Python implementation of the forecasting workflow.
* requirements.txt — list of required Python packages.
* README.md — project documentation.

🔮 Future Enhancements

* Predict humidity, rainfall, wind speed, and atmospheric pressure.
* Add historical lag features and time-series forecasting methods.
* Improve performance through hyperparameter tuning.
* Evaluate using chronological train-test splits.
* Develop a web interface using Flask or Streamlit.
* Display predictions in a weather-app-style dashboard.
* Integrate a reliable weather API for real-time forecasts.

⚠️ Limitations

* The model uses historical data and date-based features.
* Future weather cannot be determined accurately from calendar features alone.
* Predictions may not match actual weather conditions.
* Temperature-based labels are illustrative, not independently trained weather classifications.
* Model evaluation should be repeated using chronological validation before making claims about real-world forecasting accuracy.

This project is intended for educational and academic purposes. Its predictions are experimental estimates and should not be treated as official weather forecasts.

👨‍💻 Author

Mohammed Munthasir

GitHub Repository: https://github.com/mmunthasir3-oss/weather-prediction

📄 License

This project is intended for educational purposes. Check the dataset’s license and usage conditions before redistributing the dataset or its contents.