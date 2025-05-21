import sqlite3
import os

db_path = os.path.join(os.getcwd(), "face_recognition_db.sqlite")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Create tables
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    face_encoding BLOB NOT NULL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    message TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
)
""")

conn.commit()
conn.close()
print("Database initialized successfully.")
