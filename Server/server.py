
from flask import Flask
from flask_socketio import SocketIO
import ollama
import speech_recognition as sr
import face_recognition
import base64
import cv2
import numpy as np
from face_recognition_model import FaceRecognizer
from numpy.linalg import norm
import time

# Compare two face encodings
def is_same_face(enc1, enc2, threshold=0.6):
    if enc1 is None or enc2 is None:
        return False
    return norm(enc1 - enc2) < threshold


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
recognizer = sr.Recognizer()
face_db = FaceRecognizer()

# Global state variables
pending_encoding = None
pending_name = None
current_user_id = None
current_user_name = None
is_registering_user = False
last_face_encoding = None
unrecognized_start_time = None
is_listening = False


# List of supported commands
COMMAND_LIST = [
    "move forward", "move backward", "turn left", "turn right", "stop",
    "stop tracking", "track my face", "center camera",
    "play music", "honk", "take a picture"
]

@socketio.on('connect')
def handle_connect():
    print("Client connected!")

@socketio.on('robot_status')
def robot_status(data):
    print(f"Success!: {data}")
    socketio.emit('server_response', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected!")

@socketio.on('face_frame')
def handle_face_frame(data):
    """Processes incoming camera frame and detects the user"""
    global current_user_id, current_user_name, pending_encoding, is_registering_user, last_face_encoding, unrecognized_start_time, is_listening
    
    # Stop if already listening or registering a new face
    if is_listening or is_registering_user:
        return

    img_base64 = data.get('img_base64')
    if not img_base64:
        return
    # Remove base64 header if present
    if "," in img_base64:
        img_base64 = img_base64.split(",")[1]

    # Decode image from base64
    frame_bytes = base64.b64decode(img_base64)
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return
    
    # Convert to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and extract encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if not face_encodings:
        current_user_id = None
        return

    unknown_enc = face_encodings[0]

    # Try to match face to known users
    user_id, user_name = face_db.recognize_face(unknown_enc)

    # Face recognized
    if user_id:
        current_user_id = user_id
        current_user_name = user_name
        last_face_encoding = None
        unrecognized_start_time = None
        
        # Start listening if not already
        if not is_listening and not is_registering_user:
            is_listening = True
            socketio.start_background_task(record_audio)
    else:
        # Face not recognized, check for stability
        if is_same_face(unknown_enc, last_face_encoding):
            if unrecognized_start_time is None:
                unrecognized_start_time = time.time()
            # Same unknown face held steady for 3 seconds
            elif time.time() - unrecognized_start_time >= 3 and not is_registering_user:
                is_registering_user = True
                pending_encoding = unknown_enc
                socketio.emit('llm_response', {'response': "I don't recognize you. Please say your name."})
                socketio.start_background_task(capture_name_from_speech)
        else:
            # New unknown face = reset timer
            last_face_encoding = unknown_enc
            unrecognized_start_time = time.time()

def capture_name_from_speech():
    """Listens for user's name"""
    global pending_name
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            print("Listening for name...")
            # Listen for up to 5 seconds
            audio = recognizer.listen(source, timeout=5)

            # Convert speech to text
            name = recognizer.recognize_google(audio)
            pending_name = name.strip()

            # Confirm name with user
            socketio.emit('llm_response', {'response': f"Did you say {pending_name}? Please say yes or no."})
            socketio.start_background_task(capture_name_confirmation)

        except Exception as e:
            print(f"Error capturing name: {e}")
            socketio.emit('llm_response', {'response': "Sorry, I didn’t catch that. Please try again."})

def capture_name_confirmation():
    """Confirms name and registers user if correct"""
    global pending_encoding, pending_name, is_registering_user, last_face_encoding, unrecognized_start_time
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            print("Listening for confirmation...")
            audio = recognizer.listen(source, timeout=5)
            confirmation = recognizer.recognize_google(audio).lower()
            
            # If confirmed, add to DB
            if confirmation in ["yes", "yeah", "correct", "yep"]:
                face_db.register_new_user(pending_name, pending_encoding)
                socketio.emit('llm_response', {'response': f"Welcome, {pending_name}. You're now registered!"})
            else:
                socketio.emit('llm_response', {'response': "Okay, let's try again. Please say your name."})
                socketio.start_background_task(capture_name_from_speech)
        except Exception as e:
            print(f"Error confirming name: {e}")
            socketio.emit('llm_response', {'response': "Sorry, I didn’t catch that. Please try again."})
    
    # Reset registration state
    pending_encoding = None
    pending_name = None
    is_registering_user = False
    last_face_encoding = None
    unrecognized_start_time = None

def record_audio():
    """Listens to user for voice commands or conversation"""
    global current_user_id, is_registering_user, is_listening
    
    # Skip if user is not recognized or in the middle of registering
    if is_registering_user or current_user_id is None:
        print("Skipping voice input.")
        is_listening = False
        return
    

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            print("Say something...")
            # Listen for a command
            audio = recognizer.listen(source, timeout=5)

            # Convert to text
            text = recognizer.recognize_google(audio).strip()
            print("Transcription:", text)
            if text:
                get_llm_response(text)
        except Exception as e:
            print(f"Mic Error: {e}")
    
    # Done listening
    is_listening = False

def get_llm_response(text):
    """Sends user input to LLM and handles response"""
    global current_user_id, current_user_name

    # Prepare prompt for the LLM to classify the input
    prompt = f"""
    You are a robot assistant helping {current_user_name} that can understand both conversation and commands.
    Below is a list of valid commands you can execute:
    
    {COMMAND_LIST}

    Your task is to determine if the user's input is a COMMAND or a CONVERSATION.
    
    - If the input matches or resembles a command from the list above, respond with "COMMAND: <command>".
    - If the input is normal conversation, respond with "CONVERSATION: <response>".

    Example:
    User: "Can you move forward?"
    Output: COMMAND: move forward

    User: "How's the weather?"
    Output: CONVERSATION: <response>

    User Input: "{text}"
    """

    try:
        # Get LLM response
        response = ollama.chat(model="mistral:latest", messages=[{"role": "user", "content": prompt}])
        llm_response = response["message"]["content"]
        print("LLM Response:", llm_response)

         # Extract and emit command if valid
        if llm_response.startswith("COMMAND:"):
            command = llm_response.replace("COMMAND:", "").strip()
            if command in COMMAND_LIST:
                socketio.emit("execute_command", {"command": command})
            else:
                socketio.emit("llm_response", {"response": "Unrecognized command"})
        else:
            # Send normal LLM reply
            reply = llm_response.replace("CONVERSATION:", "").strip()
            socketio.emit("llm_response", {"response": reply})

        # Save conversation to user log
        if current_user_id:
            face_db.add_conversation_log(current_user_id, f"User: {text}")
            face_db.add_conversation_log(current_user_id, f"Bot: {reply}")
    except Exception as e:
        print("LLM processing failed:", e)

if __name__ == '__main__':
    print("🚀 Server is running at http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
