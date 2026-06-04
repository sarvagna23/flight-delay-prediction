import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from ingest import load_data
from sql_queries import engine, load_to_db, query_delays_by_airline, query_delays_by_month, query_delays_by_route, query_delays_by_hour

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid")

def plot_delay_by_airline(engine):
    df = query_delays_by_airline(engine)
    df = df[df['total_flights'] > 1000].sort_values('delay_rate_pct', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='delay_rate_pct', y='Airline', palette='Reds_r')
    plt.title('Flight Delay Rate by Airline (2022)', fontsize=14)
    plt.xlabel('Delay Rate (%)')
    plt.ylabel('Airline')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/delay_by_airline.png', dpi=150)
    plt.close()
    print("Saved: delay_by_airline.png")

def plot_delay_by_hour(engine):
    df = query_delays_by_hour(engine)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='hour', y='delay_rate_pct', marker='o', color='red')
    plt.fill_between(df['hour'], df['delay_rate_pct'], alpha=0.2, color='red')
    plt.title('Flight Delay Rate by Hour of Day (2022)', fontsize=14)
    plt.xlabel('Departure Hour')
    plt.ylabel('Delay Rate (%)')
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/delay_by_hour.png', dpi=150)
    plt.close()
    print("Saved: delay_by_hour.png")

def plot_delay_by_month(engine):
    df = query_delays_by_month(engine)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    df['month_name'] = df['Month'].apply(lambda x: months[x-1] if x <= 12 else x)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='month_name', y='delay_rate_pct', palette='coolwarm')
    plt.title('Flight Delay Rate by Month (2022)', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Delay Rate (%)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/delay_by_month.png', dpi=150)
    plt.close()
    print("Saved: delay_by_month.png")

def plot_top_delayed_routes(engine):
    df = query_delays_by_route(engine)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df.head(15), x='delay_rate_pct', y='route', palette='Oranges_r')
    plt.title('Top 15 Most Delayed Routes (2022)', fontsize=14)
    plt.xlabel('Delay Rate (%)')
    plt.ylabel('Route')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_delayed_routes.png', dpi=150)
    plt.close()
    print("Saved: top_delayed_routes.png")

def plot_delay_distribution(df):
    delayed = df[df['DepDel15'] == 1]['DepDelayMinutes'].dropna()
    plt.figure(figsize=(12, 6))
    sns.histplot(delayed[delayed < 200], bins=50, color='red', alpha=0.7)
    plt.title('Distribution of Delay Minutes (2022)', fontsize=14)
    plt.xlabel('Delay Minutes')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/delay_distribution.png', dpi=150)
    plt.close()
    print("Saved: delay_distribution.png")

if __name__ == "__main__":
    df = load_data()
    load_to_db(df)
    plot_delay_by_airline(engine)
    plot_delay_by_hour(engine)
    plot_delay_by_month(engine)
    plot_top_delayed_routes(engine)
    plot_delay_distribution(df)
    print("\nAll EDA charts saved to outputs/")
