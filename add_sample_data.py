import sqlite3
from datetime import datetime, timedelta
import random

# Connect to database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Names and their entry counts
entries = {
    'Nayan Patil': 12,
    'Saurav Shirkare': 19
}

# Current time
now = datetime.now()

for name, count in entries.items():
    for i in range(count):
        # Add some random minutes to spread the times
        time_offset = timedelta(minutes=random.randint(0, 1440))  # up to 24 hours ago
        entry_time = (now - time_offset).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("INSERT INTO attendance (name, time) VALUES (?, ?)", (name, entry_time))

conn.commit()
conn.close()

print("Sample attendance data added!")