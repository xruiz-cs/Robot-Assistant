import sqlite3
import numpy as np
import face_recognition
import os

# Path to the SQLite database file
DB_PATH = os.path.join(os.getcwd(), "face_recognition_db.sqlite")

class FaceRecognizer:
    def __init__(self, db_path=DB_PATH, tolerance=0.5):
        """Set up path to the database and face match tolerance"""
        self.db_path = db_path
        self.tolerance = tolerance
        self.known_faces = [] # List to store loaded (user_id, name, encoding) tuples
        self.load_known_faces()

    # Open a new database connection
    def get_connection(self):
        return sqlite3.connect(self.db_path)

    
    def load_known_faces(self):
        """Load all known face encodings from the database into memory"""
        self.known_faces.clear()
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT user_id, name, face_encoding FROM users")
        rows = cur.fetchall()
        conn.close()

        # Decode face encodings and store them in self.known_faces
        for user_id, name, enc_blob in rows:
            encoding = np.frombuffer(enc_blob, dtype=np.float64)
            self.known_faces.append((user_id, name, encoding))

    def register_new_user(self, name, face_encoding):
        """Save a new user's name and face encoding to the database"""
        enc_blob = face_encoding.tobytes()
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (name, face_encoding) VALUES (?, ?)", (name, enc_blob))
        conn.commit()
        conn.close()

        # Reload known faces after adding new one
        self.load_known_faces()
    
    def recognize_face(self, unknown_encoding):
        """Try to match the given face encoding to known faces"""
        if not self.known_faces:
            return None, None
        known_encodings = [x[2] for x in self.known_faces]

        # Compare with known encodings using given tolerance
        results = face_recognition.compare_faces(known_encodings, unknown_encoding, self.tolerance)
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        best_match = np.argmin(face_distances)

        # If best match passes the threshold, return user info
        if results[best_match]:
            user_id = self.known_faces[best_match][0]
            user_name = self.known_faces[best_match][1]
            return user_id, user_name
        return None, None

    def add_conversation_log(self, user_id, message):
        """Save a message to the conversation history for a user"""
        if user_id is None:
            return
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO conversations (user_id, message) VALUES (?, ?)", (user_id, message))
        conn.commit()
        conn.close()

    def get_user_conversation_history(self, user_id, limit=20):
        """Get recent conversation messages for a user"""
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("""SELECT timestamp, message
                       FROM conversations
                       WHERE user_id=?
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (user_id, limit))
        rows = cur.fetchall()
        conn.close()
        rows.reverse()  # Return messages oldest to newest
        return rows
