import pandas as pd
import numpy as np

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        print(f"Error: The dataset file was not found in directory '{filepath}'")
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
    return None

def clean_data(df):
    clean_df = df.copy()

    #Filling missing values with column mean
    clean_df.fillna(clean_df.mean(), inplace=True) 

    numeric_columns = clean_df.select_dtypes(include=[np.number]).columns[1:] #Exlude sample_id column from outlier check

    #Removing outliers
    for col in numeric_columns:
        std_col = df[col].std()
        mean_col = df[col].mean()
        lower_bound = mean_col - 3 * std_col
        upper_bound = mean_col + 3 * std_col
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]

    return clean_df

def compute_statistics(col,df):
    if col not in df.columns:
        print(f"Error: Column '{col}' does not exist in the dataset.")
        return None
    
    #Filling dictionary with values

    output = {
        "min" : df[col].min(),
        "max" : df[col].max(),
        "mean" : df[col].mean(),
        "median" : df[col].median(),
        "std_dev" : df[col].std(),
    }

    return output

def print_statistics(col,output):

    if output is None or col is None:
        return
    
    print(f"\nStatistics for '{col}':")
    print(f"  Minimum: {output['min']:.2f}")
    print(f"  Maximum: {output['max']:.2f}")
    print(f"  Mean: {output['mean']:.2f}")
    print(f"  Median: {output['median']:.2f}")
    print(f"  Standard Deviation: {output['std_dev']:.2f}")

def main():
    filepath = './datasets/soil_test.csv'
    df = load_data(filepath)
    if df is None:
        return
    
    clean_df = clean_data(df)
    
    for col in clean_df.columns[1:]: #Exclude sample_id column
        output = compute_statistics(col,clean_df)
        print_statistics(col, output)

if __name__ == '__main__':
    main()
