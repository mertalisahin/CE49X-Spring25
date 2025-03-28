import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    try: 
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print("File not found. Please check the directory again.")
        return None

def preprocess_data(df):
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=61)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)
    return predictions

def plot_predictions(y_test, predictions):
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.title('Actual vs. Predicted Concrete Compressive Strength')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()

def plot_residuals(y_test, predictions):
    residuals = y_test - predictions
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=predictions, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Strength (MPa)')
    plt.ylabel('Residuals (MPa)')
    plt.title('Residual Plot')
    plt.show()

def explore_data(df):
    # Display the first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Display summary statistics and data info
    print("\nDataset Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    # Check for missing values
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    df.dropna()

    for col in df.columns:
        plt.figure(figsize=(10,6))
        sns.histplot(df[col], kde=True) #kde = kernel density estimate (a method for visualizing the distribution of observations in a dataset)
        plt.title(f"Distribution of {col}")
        plt.show()

    return df

def main():
    filepath = "datasets\concrete_strength\Concrete_Data.xls"
    concrete_df = load_data(filepath)
    if concrete_df is None:
        quit()

    df = explore_data(concrete_df)
    
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)
    plot_predictions(y_test, predictions)
    plot_residuals(y_test, predictions)


if __name__ == "__main__":
    main()