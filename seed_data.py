import random
import datetime
from database.db import connect_db

def seed_attendance():
    """Generates mock attendance data for the last 60 days to improve dashboard visualization."""
    conn = connect_db()
    if not conn:
        print("Failed to connect to database.")
        return

    cursor = conn.cursor()
    
    # Names to use for mock data
    # Including registered ones and some extras for better charts
    names = ['bhanu', 'rohit', 'virat', 'shubman', 'rahul', 'pant', 'hardik', 'bumrah']
    
    # Clear existing data? (Optional, but helps for a clean visualization demo)
    # cursor.execute("TRUNCATE TABLE attendance")
    
    today = datetime.date.today()
    records_count = 0
    
    print(f"Generating data for {len(names)} people over the last 60 days...")

    for i in range(60):
        current_date = today - datetime.timedelta(days=i)
        date_str = str(current_date)
        
        # Skip Sundays (standard office scenario)
        if current_date.weekday() == 6:
            continue
            
        # Randomly decide how many people are present (between 4 and 8)
        present_count = random.randint(4, len(names))
        present_today = random.sample(names, present_count)
        
        for name in present_today:
            # Generate a random check-in time between 8:30 AM and 10:15 AM
            hour = random.randint(8, 9)
            minute = random.randint(0, 59) if hour == 9 else random.randint(30, 59)
            second = random.randint(0, 59)
            time_str = f"{hour:02d}:{minute:02d}:{second:02d}"
            
            query = "INSERT INTO attendance (name, date, time) VALUES (%s, %s, %s)"
            cursor.execute(query, (name, date_str, time_str))
            records_count += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Successfully inserted {records_count} mock records!")

if __name__ == "__main__":
    seed_attendance()
