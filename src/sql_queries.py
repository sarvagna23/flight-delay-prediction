import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = "sqlite:///./flight_data.db"
engine = create_engine(DATABASE_URL)

def load_to_db(df: pd.DataFrame):
    print("Loading data to SQLite...")
    # Load sample for SQL queries (100K rows)
    df.head(100000).to_sql('flights', engine, if_exists='replace', index=False)
    print("Done — 100,000 records loaded to DB")

def query_delays_by_airline(engine) -> pd.DataFrame:
    query = text("""
        SELECT 
            Airline,
            COUNT(*) as total_flights,
            SUM(DepDel15) as delayed_flights,
            ROUND(AVG(DepDel15) * 100, 2) as delay_rate_pct,
            ROUND(AVG(DepDelayMinutes), 2) as avg_delay_minutes
        FROM flights
        WHERE Cancelled = 0
        GROUP BY Airline
        ORDER BY delay_rate_pct DESC
    """)
    return pd.read_sql(query, engine)

def query_delays_by_month(engine) -> pd.DataFrame:
    query = text("""
        SELECT 
            Month,
            COUNT(*) as total_flights,
            ROUND(AVG(DepDel15) * 100, 2) as delay_rate_pct,
            ROUND(AVG(DepDelayMinutes), 2) as avg_delay_minutes
        FROM flights
        WHERE Cancelled = 0
        GROUP BY Month
        ORDER BY Month
    """)
    return pd.read_sql(query, engine)

def query_delays_by_route(engine) -> pd.DataFrame:
    query = text("""
        SELECT 
            Origin || ' -> ' || Dest as route,
            COUNT(*) as total_flights,
            ROUND(AVG(DepDel15) * 100, 2) as delay_rate_pct
        FROM flights
        WHERE Cancelled = 0
        GROUP BY Origin, Dest
        HAVING total_flights > 100
        ORDER BY delay_rate_pct DESC
        LIMIT 20
    """)
    return pd.read_sql(query, engine)

def query_delays_by_hour(engine) -> pd.DataFrame:
    query = text("""
        SELECT 
            CAST(CRSDepTime / 100 AS INTEGER) as hour,
            COUNT(*) as total_flights,
            ROUND(AVG(DepDel15) * 100, 2) as delay_rate_pct
        FROM flights
        WHERE Cancelled = 0
        GROUP BY hour
        ORDER BY hour
    """)
    return pd.read_sql(query, engine)

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from ingest import load_data
    
    df = load_data()
    load_to_db(df)
    
    print("\n--- Delays by Airline ---")
    print(query_delays_by_airline(engine).head(10).to_string())
    
    print("\n--- Delays by Month ---")
    print(query_delays_by_month(engine).to_string())
    
    print("\n--- Top Delayed Routes ---")
    print(query_delays_by_route(engine).head(10).to_string())
    
    print("\n--- Delays by Hour ---")
    print(query_delays_by_hour(engine).to_string())