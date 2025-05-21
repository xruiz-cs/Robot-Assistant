# AI-Powered Robot Assistant

This project is a real-time, modular robot assistant built on the PiCar-X platform using a Raspberry Pi. It performs facial recognition, speech recognition, and real-time interaction through a local API and WebSocket connection with a desktop server running an LLM.

---

## Overview

### Server (desktop): 
Handles facial recognition, voice processing, user registration, and LLM interaction.

### Client (PiCar-X): 
Streams video, executes movement/camera commands, and speaks responses via TTS.


---

## Server Structure (`server/`)
* `server.py`: Flask-SocketIO server that receives video frames, detects users, and interacts with the LLM.
* `face_recognition_model.py`: Contains `FaceRecognizer` class for face matching, registration, and storing/retrieving user conversations.
* `init_face_db.py`: Initializes the SQLite database with `users` and `conversations` tables.

--- 

## Client Structure (`client/` on Raspberry Pi)
* `client.py`: Starts video streaming, connects to the server, executes incoming commands, and handles TTS responses.
* `face_tracking.py`: Tracks and centers the user’s face using OpenCV and camera servos.
* `robot_commands.py`: Contains motor, servo, and utility command implementations

---

## Setup

### Server Setup

1. Install dependencies:

   ```bash
   pip install flask flask-socketio face_recognition speechrecognition numpy opencv-python ollama
   ```
2. Run the database initializer:

   ```bash
   python init_face_db.py
   ```
3. Start the server:

   ```bash
   python server.py
   ```

### Client Setup

1. Install required packages:
Install PiCar-X dependencies (see SunFounder docs).

   ```bash
   pip install opencv-python numpy socketio-client
   ```

2. Update the IP in `client.py`:

   ```python
   SERVER_IP = "your-desktop-ip-address"
   ```

3. Start the client:

   ```bash
   sudo python client.py
   ```

---

## Supported Voice Commands

* move forward
* move backward
* turn left / turn right
* stop
* track my face / stop tracking
* center camera
* take a picture
* honk / play music

---

## How It Works

1. User appears in front of the camera. The server attempts to recognize the face.
2. If not recognized, the assistant asks for the user's name and registers the face.
3. The assistant listens for voice input, processes it via an LLM, and classifies it as either a command or a conversation.
4. Commands are executed on the PiCar-X. Conversations are logged and replied to via TTS.

---

## Notes

* Be sure to run the PiCar-X side with `sudo` to allow hardware access and TTS output.
* Facial encodings and conversation history are stored in `face_recognition_db.sqlite`.
* System is designed for LAN usage with minimal latency.
