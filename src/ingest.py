import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Combined_Flights_2022.csv')

def load_data(path: str = DATA_PATH, sample: bool = True) -> pd.DataFrame:
    print("Loading flight data...")
    if sample:
        # Load 2M rows for performance
        df = pd.read_csv(path, nrows=2000000)
    else:
        df = pd.read_csv(path)
    print(f"Loaded {len(df):,} records")
    print(f"Columns: {list(df.columns)}")
    return df

def validate_data(df: pd.DataFrame) -> bool:
    print("\nValidating data...")
    print(f"Shape: {df.shape}")
    print(f"Null values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Delayed flights: {df['DepDel15'].sum():,} ({df['DepDel15'].mean()*100:.2f}%)")
    return True

if __name__ == "__main__":
    df = load_data()
    validate_data(df)