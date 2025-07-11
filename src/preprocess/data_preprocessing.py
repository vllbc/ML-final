import pandas as pd
import numpy as np
import os

def preprocess_data(df, is_train=True, fit_scaler=None):
    """
    Preprocesses the raw data by handling missing values and resampling.
    """
    # Use provided column names if it's test data
    if not is_train:
        df.columns = ['DateTime','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    # Convert DateTime to datetime objects and set as index
    # The format in test.csv might be different, let's unify it
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Replace '?' with NaN and convert columns to numeric
    df.replace('?', np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Forward fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True) # bfill to handle NaNs at the beginning
    
    # Resample to daily frequency
    daily_df = df.resample('D').agg({
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }).fillna(0) # Fill any remaining NaNs after resampling

    # Create the remainder sub-metering feature
    daily_df['sub_metering_remainder'] = (daily_df['Global_active_power'] * 1000 / 60) - \
                                         (daily_df['Sub_metering_1'] + 
                                          daily_df['Sub_metering_2'] + 
                                          daily_df['Sub_metering_3'])
    
    return daily_df

def main():
    """
    Main function to run the data preprocessing pipeline for train and test sets separately.
    """
    # Define file paths
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    output_train_path = 'data/processed_train.csv'
    output_test_path = 'data/processed_test.csv'

    # --- Process Training Data ---
    print("Loading and preprocessing training data...")
    train_df_raw = pd.read_csv(train_path, sep=',')
    processed_train_df = preprocess_data(train_df_raw, is_train=True)
    
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    processed_train_df.to_csv(output_train_path)
    print(f"Processed training data saved to {output_train_path}")

    # --- Process Test Data ---
    print("\nLoading and preprocessing testing data...")
    test_df_raw = pd.read_csv(test_path, sep=',', header=None)
    processed_test_df = preprocess_data(test_df_raw, is_train=False)

    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
    processed_test_df.to_csv(output_test_path)
    print(f"Processed testing data saved to {output_test_path}")

    print("\nData preprocessing complete.")

if __name__ == '__main__':
    main() 