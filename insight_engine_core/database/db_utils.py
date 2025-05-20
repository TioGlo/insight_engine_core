# insight_engine_core/database/db_utils.py
# Import Base, engine from the __init__.py in the same directory
from . import Base, engine  # Or from .__init__ import Base, engine (less common)

# from ..config import DATABASE_URL # No longer needed here if __init__ handles it

print(f"db_utils.py: Imported Base object with id: {id(Base)} from database/__init__.py")


def init_db():
    print(f"init_db(): Using Base object with id: {id(Base)}")
    print("init_db(): Attempting to import models...")

    # This import will cause models.py to run.
    # models.py will then import Base from . (i.e., from database/__init__.py)
    from . import models

    print(f"init_db(): After models import, tables known to Base.metadata: {Base.metadata.tables.keys()}")

    if not Base.metadata.tables:
        print("init_db(): WARNING - No tables found in Base.metadata. Check model definitions and Base inheritance.")
        return

    try:
        print("init_db(): Calling Base.metadata.create_all(bind=engine)...")
        Base.metadata.create_all(bind=engine)  # engine is also imported from .
        print("init_db(): Base.metadata.create_all(bind=engine) executed.")
    except Exception as e:
        print(f"init_db(): An error occurred during create_all: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("Running db_utils.py directly for init_db()")
    init_db()