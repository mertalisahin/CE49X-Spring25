# lab1_traffic_analysis.py

import pandas as pd

def main():
    # Load the traffic dataset
    try:
        df = pd.read_csv('./datasets/traffic_data.csv')
    except FileNotFoundError:
        print("Error: The dataset file was not found. Please ensure 'traffic_data.csv' is located in the /datasets/ folder.")
        return

    # Compute basic descriptive statistics for 'vehicle_count'
    min_value = df['vehicle_count'].min()
    max_value = df['vehicle_count'].max()
    mean_value = df['vehicle_count'].mean()

    # Print the results
    print("Traffic Data Analysis:")
    print(f"Minimum vehicle count: {min_value}")
    print(f"Maximum vehicle count: {max_value}")
    print(f"Mean vehicle count: {mean_value:.2f}")

if __name__ == '__main__':
    main()