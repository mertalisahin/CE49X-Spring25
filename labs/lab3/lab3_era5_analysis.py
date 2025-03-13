import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_explore_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading {file_path}")
        return None

    # Convert timestamp to datetime
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Error converting timestamp in {file_path}")

    # Display basic information
    print(f"\nDataset: {file_path}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Data Types:\n", df.dtypes)
    
    # Handling missing values
    if df.isnull().values.any():
        df.fillna(df.mean(), inplace=True)
    # Display summary statistics
    print("Summary Statistics:\n", df.describe())
    return df

def calculate_wind_speed(df):
    df['wind_speed'] = np.sqrt(df['u10m']**2 + df['v10m']**2)
    return df
def get_season(month):

    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Unknown'

def add_season_column(df):
    df['season'] = df['timestamp'].dt.month.apply(get_season)
    return df

def aggregate_monthly(df, value_col):
    df.set_index('timestamp', inplace=True)
    monthly_avg = df[value_col].resample('ME').mean()
    df.reset_index(inplace=True)
    return monthly_avg

def aggregate_seasonal(df, value_col):
    df = add_season_column(df)
    seasonal_avg = df.groupby('season')[value_col].mean()
    return seasonal_avg

def extreme_weather_periods(df, quantile=0.98):
    threshold = df['wind_speed'].quantile(quantile)
    extreme_df = df[df['wind_speed'] > threshold].sort_values(by='wind_speed', ascending=False)
    return extreme_df


def diurnal_pattern(df, value_col):
    df['hour'] = df['timestamp'].dt.hour
    diurnal_avg = df.groupby('hour')[value_col].mean()
    return diurnal_avg

def plot_monthly_wind_speed(berlin_monthly, munich_monthly):
    plt.figure(figsize=(10, 5))
    plt.plot(berlin_monthly.index, berlin_monthly.values, label="Berlin", marker='o')
    plt.plot(munich_monthly.index, munich_monthly.values, label="Munich", marker='s')
    plt.xlabel("Month")
    plt.ylabel("Average Wind Speed (m/s)")
    plt.title("Monthly Average Wind Speeds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_seasonal_comparison(berlin_seasonal, munich_seasonal):

    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    berlin_values = [berlin_seasonal.get(season, np.nan) for season in seasons]
    munich_values = [munich_seasonal.get(season, np.nan) for season in seasons]
    
    x = np.arange(len(seasons))
    width = 0.35  # width of the bars
    
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, berlin_values, width, label='Berlin')
    plt.bar(x + width/2, munich_values, width, label='Munich')
    plt.xticks(x, seasons)
    plt.xlabel("Season")
    plt.ylabel("Average Wind Speed")
    plt.title("Seasonal Comparison of Wind Speeds")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_wind_direction(df, city_name="City"):
    

    wind_dir = (np.degrees(np.arctan2(df['v10m'], df['u10m'])) + 360) % 360


    bins = np.arange(0, 361, 22.5)
    counts, _ = np.histogram(wind_dir, bins=bins)

    theta = np.deg2rad(bins[:-1] + 11.25)  # center of each bin
    widths = np.deg2rad([22.5] * len(theta))
    
    ax = plt.subplot(111, polar=True)
    ax.bar(theta, counts, width=widths, bottom=0.0, edgecolor='black', align='center')
    plt.title(f"Wind Direction Distribution in {city_name}")
    plt.tight_layout()
    plt.show()

def main():

    berlin_file = './datasets/berlin_era5_wind_20241231_20241231.csv'
    munich_file = './datasets/munich_era5_wind_20241231_20241231.csv'
    
    # Load and explore datasets
    berlin_df = load_and_explore_dataset(berlin_file)
    munich_df = load_and_explore_dataset(munich_file)
    
    if berlin_df is None or munich_df is None:
        print("One or both datasets could not be loaded. Exiting.")
        quit()
    
    # Calculate wind speed for both datasets
    berlin_df = calculate_wind_speed(berlin_df)
    munich_df = calculate_wind_speed(munich_df)
    
    # Compute monthly averages for wind speed
    berlin_monthly_wind = aggregate_monthly(berlin_df.copy(), 'wind_speed')
    munich_monthly_wind = aggregate_monthly(munich_df.copy(), 'wind_speed')
    
    # Compute seasonal averages for wind speed
    berlin_seasonal_wind = aggregate_seasonal(berlin_df.copy(), 'wind_speed')
    munich_seasonal_wind = aggregate_seasonal(munich_df.copy(), 'wind_speed')
    
    # Identify extreme weather periods
    print("\nBerlin Extreme Weather Periods:")
    print(extreme_weather_periods(berlin_df))
    
    print("\nMunich Extreme Weather Periods:")
    print(extreme_weather_periods(munich_df))
    
    # Calculate diurnal patterns
    berlin_diurnal = diurnal_pattern(berlin_df.copy(), 'wind_speed')
    munich_diurnal = diurnal_pattern(munich_df.copy(), 'wind_speed')
    print("\nBerlin Diurnal Wind Speed Pattern:\n", berlin_diurnal)
    print("\nMunich Diurnal Wind Speed Pattern:\n", munich_diurnal)
    
    # Visualizations

    plot_monthly_wind_speed(berlin_monthly_wind, munich_monthly_wind)
    
    plot_seasonal_comparison(berlin_seasonal_wind, munich_seasonal_wind)
    
    plot_wind_direction(berlin_df, city_name="Berlin")
    plot_wind_direction(munich_df, city_name="Munich")

    
if __name__ == "__main__":
    main()
