# 🔥 Employee Burnout Analysis Dashboard

A comprehensive web application for analyzing and predicting employee burnout rates using machine learning.

## Features

- **📊 Overview Dashboard**: Key metrics and dataset statistics
- **📈 Interactive Visualizations**: 
  - Correlation heatmaps
  - Distribution charts for all features
  - Box plots and scatter plots
  - Interactive Plotly charts
- **🤖 Burnout Predictor**: Real-time burnout rate prediction based on employee attributes
- **📋 Data Explorer**: Filter and explore the dataset with download capabilities

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the data file is present**:
   - Make sure `employee_burnout_analysis-AI.xlsx` is in the project directory

## Usage

1. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the dashboard**:
   - The application will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

## Application Pages

### 📊 Overview
- View dataset statistics and key metrics
- Check data quality (missing values, data types)
- See model performance metrics

### 📈 Visualizations
- Explore various data visualizations
- Interactive charts with Plotly
- Analyze relationships between features and burnout rate

### 🤖 Predict Burnout
- Input employee details (Designation, Resource Allocation, Mental Fatigue Score, Gender, Company Type, WFH Setup)
- Get instant burnout rate predictions
- Receive risk level interpretations

### 📋 Data Explorer
- Filter data by multiple criteria
- View and download filtered datasets
- Explore raw employee data

## Model Information

- **Algorithm**: Random Forest Regressor
- **Feature Engineering**: PCA (Principal Component Analysis) with 95% variance retention
- **Training Accuracy**: ~91%
- **Test Accuracy**: ~84%

## Features Used for Prediction

1. Designation
2. Resource Allocation
3. Mental Fatigue Score
4. Gender (encoded)
5. Company Type (encoded)
6. WFH Setup Available (encoded)

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## Notes

- The application uses caching to improve performance
- All visualizations are interactive and can be zoomed/explored
- Predictions are based on a pre-trained Random Forest model

## License

This project is for educational and analysis purposes.

