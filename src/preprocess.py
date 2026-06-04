import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from ingest import load_data

def preprocess(df: pd.DataFrame):
    print("Preprocessing data...")

    # Drop cancelled/diverted flights
    df = df[(df['Cancelled'] == 0) & (df['Diverted'] == 0)].copy()

    # Drop rows with missing target
    df = df.dropna(subset=['DepDel15'])

    # Feature engineering
    df['hour'] = (df['CRSDepTime'] // 100).astype(int)
    df['is_weekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)
    df['is_evening'] = (df['hour'] >= 18).astype(int)
    df['is_morning'] = (df['hour'] <= 9).astype(int)
    df['is_peak_hour'] = df['hour'].isin([7, 8, 17, 18, 19]).astype(int)

    # Historical delay rates per airline
    airline_delay_rate = df.groupby('Airline')['DepDel15'].mean().rename('airline_delay_rate')
    df = df.join(airline_delay_rate, on='Airline')

    # Historical delay rates per origin airport
    origin_delay_rate = df.groupby('Origin')['DepDel15'].mean().rename('origin_delay_rate')
    df = df.join(origin_delay_rate, on='Origin')

    # Historical delay rates per dest airport
    dest_delay_rate = df.groupby('Dest')['DepDel15'].mean().rename('dest_delay_rate')
    df = df.join(dest_delay_rate, on='Dest')

    # Historical delay rates per route
    route_delay_rate = df.groupby(['Origin', 'Dest'])['DepDel15'].mean().rename('route_delay_rate')
    df = df.join(route_delay_rate, on=['Origin', 'Dest'])

    # Encode airline
    le = LabelEncoder()
    df['Airline_encoded'] = le.fit_transform(df['Airline'].astype(str))

    # Select features
    features = [
        'Month', 'DayOfWeek', 'DayofMonth', 'hour',
        'Distance', 'DistanceGroup', 'Quarter',
        'is_weekend', 'is_evening', 'is_morning', 'is_peak_hour',
        'OriginAirportID', 'DestAirportID', 'Airline_encoded',
        'airline_delay_rate', 'origin_delay_rate',
        'dest_delay_rate', 'route_delay_rate'
    ]

    # Fill nulls
    df[features] = df[features].fillna(0)

    X = df[features]
    y = df['DepDel15'].astype(int)

    print(f"Features: {len(features)} total")
    print(f"Dataset size: {len(X):,}")
    print(f"Delay rate: {y.mean()*100:.2f}%")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_res):,}")

    return X_train_res, X_test, y_train_res, y_test, features

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, features = preprocess(df)
    print("Preprocessing complete")