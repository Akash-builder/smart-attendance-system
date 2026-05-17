import sys
import os
import sqlite3
from config import DB_PATH
from pathlib import Path

# Setup for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def connect_db():
    """Establish connection to SQLite database"""
    try:
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"Error connecting to SQLite: {e}")
        return None

def create_table():
    """Initialize the database table if needed"""
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                date TEXT,
                time TEXT
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()

def insert_attendance(name, date, time):
    """Save a new recognition entry into the database"""
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        query = "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)"
        cursor.execute(query, (name, date, time))
        conn.commit()
        cursor.close()
        conn.close()

def fetch_all():
    """Get every record in the table"""
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance")
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    return []

def fetch_by_date(day):
    """Filter records by a specific date"""
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE date = ?", (day,))
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    return []

# Initialize database on startup
if __name__ == "__main__":
    create_table()
else:
    # Auto-create table when imported by dashboard
    try:
        create_table()
    except:
        pass