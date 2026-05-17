from database.db import fetch_all

# Simple script to print database contents
records = fetch_all()

print("Attendance Records currently in Database:")
print("-" * 40)

for row in records:
    print(row)

if not records:
    print("Database is currently empty.")