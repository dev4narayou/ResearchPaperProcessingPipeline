from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def test_db_connection(db_url: str) -> bool:
    """
    Try to connect to the database and run `SELECT 1`.
    Returns True if successful, otherwise prints the error and returns False.
    """
    try:
        # `echo=True` logs the queries so you can see exactly what happens
        engine = create_engine(db_url, echo=False, pool_pre_ping=True)

        # `with engine.connect()` both opens and closes the connection safely
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Connection succeeded")
        return True

    except SQLAlchemyError as err:
        print("Connection failed:", err)
        return False

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    print(url)
    test_db_connection(url)