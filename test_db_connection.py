import psycopg2
from config.config import settings

def test_connection():
    # try both localhost and container name
    urls = [
        settings.DATABASE_URL,
        settings.DATABASE_URL.replace('localhost', 'ResearchAgent-pg-db')
    ]

    for url in urls:
        print(f"\nAttempting to connect to: {url}")
        try:
            conn = psycopg2.connect(url)
            print("Successfully connected to the database!")
            cur = conn.cursor()
            cur.execute("SELECT version();")
            version = cur.fetchone()
            print(f"PostgreSQL version: {version[0]}")
            cur.close()
            conn.close()
            return
        except Exception as e:
            print(f"Error connecting to the database: {e}")

if __name__ == "__main__":
    test_connection()