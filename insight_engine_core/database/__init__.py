# insight_engine_core/database/__init__.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from ..config import DATABASE_URL # Go up one level to insight_engine_core.config

print(f"database/__init__.py: Using DATABASE_URL: {DATABASE_URL}")

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define THE SINGLE Base instance here
Base = declarative_base()
print(f"database/__init__.py: Defined Base object with id: {id(Base)}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
