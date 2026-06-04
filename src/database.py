from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
import uuid

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./flight_predictions.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class FlightPrediction(Base):
    __tablename__ = "flight_predictions"
    id = Column(String, primary_key=True)
    month = Column(Integer)
    day_of_week = Column(Integer)
    hour = Column(Integer)
    airline = Column(String)
    origin = Column(String)
    dest = Column(String)
    distance = Column(Float)
    is_delayed = Column(Integer)
    delay_probability = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def log_prediction(month, day_of_week, hour, airline, origin, dest, distance, is_delayed, prob):
    session = SessionLocal()
    try:
        record = FlightPrediction(
            id=str(uuid.uuid4()),
            month=month,
            day_of_week=day_of_week,
            hour=hour,
            airline=airline,
            origin=origin,
            dest=dest,
            distance=distance,
            is_delayed=is_delayed,
            delay_probability=prob
        )
        session.add(record)
        session.commit()
    finally:
        session.close()