import sys
import requests
import speech_recognition as sr
import re
import pyttsx3
import subprocess
import os
import torch
import random
import pygame
import psutil
from googlesearch import search
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import time
import cv2
import numpy as np
import threading
import sqlite3
import schedule
from transformers import pipeline
from datetime import datetime
import datefinder
from sympy.polys.polyconfig import query

import nmap
import socket
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer



# Load YOLO model for object detection (YOLOv3)
yolo_net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]


# Load the DistilBERT tokenizer and model for intent classification
intent_model = DistilBertForSequenceClassification.from_pretrained("./intent_classifier1")
intent_tokenizer = DistilBertTokenizer.from_pretrained("./intent_classifier1")

# Load the GPT-2 model and tokenizer
def load_model():
    model = GPT2LMHeadModel.from_pretrained("./jarvismodeltrial1")
    tokenizer = GPT2Tokenizer.from_pretrained("./jarvismodeltrial1")
    tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to the EOS token
    return model, tokenizer

# Initialize model and tokenizer
gpt2_model, gpt2_tokenizer = load_model()

# Function to generate a response using GPT-2
import re

def generate_response(query):
    # Encode the query and generate a response
    inputs = gpt2_tokenizer.encode(query, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
    response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove unwanted tags like [Emotion: ...], [Style: ...], [Type: ...]
    cleaned_response = re.sub(r"\[Emotion:.*?\]|\[Style:.*?\]|\[Type:.*?\]", "", response).strip()

    # Remove the query from the response if it appears at the beginning
    if cleaned_response.startswith(query):
        cleaned_response = cleaned_response[len(query):].strip()

    return cleaned_response


def speak_text_async(text):
    def speak():
        tts_engine.say(text)
        tts_engine.runAndWait()


    threading.Thread(target=speak).start()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Initialize Pygame Mixer for Music
pygame.mixer.init()


# Global Variables
keyword = "jarvis"  # Keyword to activate the assistant
chat_history_ids = None
music_folder = "C:/Users/ASUS/Music/christmass"  # Replace with your music folder path
music_files = []  # Will be populated with music file paths
current_track_index = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
song_path = "C:/Users/ASUS/Music/santa-papa-noel-feliz-navidad-01-127494.mp3"


# Function to classify intent using DistilBERT
# Confidence threshold to activate the functions
CONFIDENCE_THRESHOLD = 0.21517860412597656


def classify_intent(user_input):
    inputs = intent_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = intent_model(**inputs).logits
        softmax_probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = softmax_probs[0][predicted_class].item()
        print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

        # Only return intent if confidence is above threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            return predicted_class, confidence
        else:
            print("Intent confidence is too low, no action taken.")
            return None, confidence


# Map intent classes to functions
intent_map = {
    0: "play music",  # Play the next track, start playback, etc.
    1: "stop music",  # Stop the current music or pause playback
    2: "open application",  # Launch applications like Notepad
    3: "search internet",  # Search for information online
    4: "chat",              # General chat queries or informational responses
    5: "start detection",
    6: "stop detection",
7: "set reminder",
    8: "check reminders",
    9: "reschedule reminder",
    10: "list reminders",

}



# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Initialize the face recognizer (LBPH)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Folder path for face images
faces_dir = 'C:/Users/ASUS/Desktop/faces'
# Train the face recognizer with images from the 'faces' folder
def train_face_recognizer():
    faces = []
    labels = []
    label_map = {}

    for label, person_name in enumerate(os.listdir(faces_dir)):
        person_path = os.path.join(faces_dir, person_name)
        if os.path.isdir(person_path):
            label_map[label] = person_name
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces.append(image)
                labels.append(label)

    # Train the recognizer
    face_recognizer.train(faces, np.array(labels))
    return label_map

# Load the trained face recognizer
label_map = train_face_recognizer()

# Initialize global variables
is_detection_active = False
detected_people = {}


# Global Variables
should_speak = True  # Flag to control speaking


# This function will be responsible for periodic speaking
def periodic_speech():
    global should_speak
    while True:
        time.sleep(30)  # Wait for 30 seconds
        if should_speak and "godwin" in detected_people:
            speak_text_async("Hello Sir.")  # Speak "Hello Sir" every 30 seconds
        else:
            time.sleep(1)  # Check every second if the assistant should speak


# Modified detect_faces_and_objects function
def detect_faces_and_objects(frame):
    """Detect faces and objects in the frame and handle their identification"""
    global detected_people

    # Resize frame to feed into YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    # Process object detection results
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Draw rectangle and label
                cv2.rectangle(frame, (center_x, center_y), (center_x + w, center_y + h), (0, 255, 0), 2)
                label = str(class_id)
                cv2.putText(frame, label, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Identify "person" class
                if label == "person":
                    detected_people["person"] = "visitor"
                    speak_text_async("Sir, you have a visitor")

    # Face Detection using OpenCV's Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_image = gray[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(face_image)

        if confidence < 100:
            person_name = label_map.get(label, "Unknown")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Hello {person_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Recognize family or personal faces
            if person_name == "me":
                detected_people["me"] = "known_person"
                speak_text_async("Hello, Sir. It's you!")
            elif person_name == "family":
                detected_people["family"] = "known_family"
                speak_text_async("Hello, family member!")
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Global variables for face detection
is_detection_active = False
detection_thread = None
detection_event = threading.Event()
detection_lock = threading.Lock()


# Global variables for face detection
is_detection_active = False
detection_event = threading.Event()  # Event to control detection thread
detection_thread = None
detection_lock = threading.Lock()

def start_detection():
    """Start face and object detection loop with continuous passive listening"""
    global is_detection_active
    with detection_lock:
        if is_detection_active:  # Check if detection is already running
            return  # Exit if it's already running
        is_detection_active = True

    capture = cv2.VideoCapture(0)  # Use the first camera

    while is_detection_active:
        ret, frame = capture.read()
        if not ret:
            break

        # Simulate face and object detection
        frame = detect_faces_and_objects(frame)  # Your detection logic here

        # Show the frame with detection
        cv2.imshow("Face and Object Detection", frame)

        # Check if we should stop detection (via event flag)
        if detection_event.is_set():  # If stop event is set, break the loop
            break

        # Stop detection when 'q' is pressed (this is a safeguard)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

    with detection_lock:
        is_detection_active = False


def stop_detection():
    """Stop face and object detection"""
    global is_detection_active
    with detection_lock:
        if not is_detection_active:  # If detection is not active, do nothing
            return
        is_detection_active = False
    detection_event.set()  # Set the event to stop the detection loop
    if detection_thread is not None:
        detection_thread.join()  # Wait for the detection thread to finish
    cv2.destroyAllWindows()



# Function to load music files from a folder
def load_music(folder):
    global music_files
    music_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith((".mp3", ".wav", ".mpeg"))]
    if not music_files:
        print("No music files found in the specified folder.")
        speak_text("No music files found in the specified folder.")
    else:
        print(f"Loaded {len(music_files)} music files.")
        random.shuffle(music_files)  # Shuffle music for variety


from datetime import datetime, timedelta
import datefinder


# Function to play music
def play_music():
    global current_track_index
    if music_files:
        track = music_files[current_track_index]
        print(f"Playing: {os.path.basename(track)}")
        speak_text(f"Now playing {os.path.basename(track)}")
        pygame.mixer.music.load(track)
        pygame.mixer.music.play()
    else:
        speak_text("No music files are loaded. Please check the folder.")


# Function to stop music
def stop_music():
    pygame.mixer.music.stop()
    print("Music stopped.")
    speak_text("Music has been stopped.")


# Function to play the next track
def next_track():
    global current_track_index
    if music_files:
        current_track_index = (current_track_index + 1) % len(music_files)
        play_music()
    else:
        speak_text("No music files are available to play the next track.")

# Function to convert text to speech
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


# Function to perform Google search
def search_internet(query):
    try:
        speak_text(f"Searching the internet for {query}")
        print(f"Searching for: {query}")
        results = search(query, num_results=3)  # Get top 3 search results
        if results:
            first_result = results[0]
            speak_text(f"Opening {first_result} in Chrome")
            subprocess.Popen([r"C:\Program Files\Google\Chrome\Application\chrome.exe", first_result])
            return "I have opened the first result in Google Chrome."
        else:
            return "Sorry, I couldn't find any results."
    except Exception as e:
        print(f"Error during internet search: {e}")
        return "I encountered an error while searching the internet."


# Function to open applications
def open_application(app_name):
    try:
        app_paths = {
            "notepad": "notepad.exe",
            "google chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "calculator": "calc.exe",
            "file explorer": "explorer.exe"
        }

        if app_name in app_paths:
            subprocess.Popen(app_paths[app_name], shell=True)
            speak_text(f"Opening {app_name}")
        else:
            speak_text(f"Sorry, I don't know how to open {app_name}")
    except Exception as e:
        print(f"Error opening application: {e}")
        speak_text("I encountered an error while trying to open the application.")


# Function to close applications
def close_application(app_name):
    try:
        app_names = {
            "notepad": "notepad.exe",
            "google chrome": "chrome.exe",
            "calculator": "calc.exe",
            "file explorer": "explorer.exe"
        }

        if app_name in app_names:
            os.system(f"taskkill /f /im {app_names[app_name]}")
            speak_text(f"Closing {app_name}")
        else:
            speak_text(f"Sorry, I don't know how to close {app_name}")
    except Exception as e:
        print(f"Error closing application: {e}")
        speak_text("I encountered an error while trying to close the application.")


# Function to fetch news based on the query
def fetch_news(query):
    # Your NewsAPI API key
    API_KEY = 'fc5379610a5f49d7916bf3b48983cae6'
    # Construct the API URL with the user's query
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}'


    response = requests.get(url)

    if response.status_code == 200:
        news_data = response.json()
        articles = news_data['articles']

        if articles:
            speak_text(f"Here are the latest news articles about {query}:")
            for article in articles[:5]:  # Read the top 5 articles
                title = article['title']
                description = article['description']
                speak_text(f"Title: {title}")
                speak_text(f"Description: {description}")
        else:
            speak_text(f"Sorry, I couldn't find any articles about {query}.")
    else:
        speak_text(f"Sorry, I encountered an error: {response.status_code}")


# Function to check battery status
def check_battery():
    battery = psutil.sensors_battery()
    if battery:
        percent = battery.percent
        plugged = battery.power_plugged
        if plugged:
            return f"The battery is at {percent}% and is currently charging."
        else:
            return f"The battery is at {percent}%. It's not charging."
    else:
        return "I couldn't retrieve the battery status."


# Function to handle user query
# Function to handle user query based on intent
def handle_query(query):
    global current_track_index
    print(f"User Query: {query}")
    try:
        # Classify the intent
        intent, confidence = classify_intent(query)

        # If intent is None or confidence is below threshold, return
        if intent is None:
            speak_text("I'm not confident enough to understand your query. Please try again.")
            return

        # Map intent to corresponding action
        if intent == 0:  # Play music

            speak_text("Playing the next track.")
            load_music(music_folder)
            play_music()

        elif intent == 1:  # Stop music

            speak_text("Stopping the music.")
            stop_music()

        elif intent == 2:  # Open application

            app_name = query.lower().replace("open", "").strip()
            speak_text(f"Opening {app_name}.")
            open_application(app_name)

        elif intent == 3:  # Search internet
            search_query = query.lower().replace("search", "").strip()
            response = search_internet(search_query)
            print(f"Search Result: {response}")
            speak_text(response)

        elif intent == 4:  # Chat
             #fetch_news(query)
             response = generate_response(query)
             print(f"AI Response: {response}")
             speak_text(response)
        elif intent == 5:
            speak_text("Starting face and object detection.")
            # Start detection in a separate thread if it's not already running
            if not is_detection_active:
                detection_event.clear()  # Clear the stop signal
                detection_thread = threading.Thread(target=start_detection)
                detection_thread.start()
        elif intent == 6:
            speak_text("Stopping face and object detection.")
            # Stop detection if it's active
            stop_detection()
        elif "stop saying and leave it on the screen" in query.lower():
            global should_speak
            should_speak = False
            speak_text("Stopping voice notifications. I will only print the results now.")
            print("Voice notifications stopped.")
        elif "you can sleep" in query.lower():
            sys.exit()
        elif intent==7:
            time_in_words = get_time_in_words()
            print(time_in_words)
            speak_text(time_in_words)
        elif intent==8:
            diagnostics = system_diagnostics()
            diagnostic_report = "\n".join([f"{k}: {v}" for k, v in diagnostics.items()])
            print(diagnostic_report)
            speak_text(f"System diagnostics completed. Here is a summary. {diagnostic_report}")

        elif intent==9:
            speak_text("Scanning the network, this may take a moment.")
            scan_results = network_scan("192.168.1.0/24")

            if isinstance(scan_results, str):
                print(scan_results)
                speak_text(scan_results)
            else:
                report = "\n".join(
                    [f"Device: {ip}, Open Ports: {info['Open Ports']}, OS: {info['OS Guess']}" for ip, info in
                     scan_results.items()])
                print(report)
                speak_text(f"Network scan complete. Found {len(scan_results)} active devices.")

        elif intent==10:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                speak_text("Please say the URL or IP address you want to scan.")
                print("Listening for a URL or IP address to scan...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)



        else:
            speak_text("Sorry, I couldn't understand the intent of your query.")

    except Exception as e:
        print(f"Error processing query: {e}")
        speak_text("Sorry, I couldn't process that.")





# Function to handle passive listening
def passive_listen(recognizer, mic):
    print("Listening for the keyword...")
    try:
        audio = recognizer.listen(mic, timeout=None, phrase_time_limit=7)
        detected_text = recognizer.recognize_google(audio).lower()
        print(f"Passive Listening Heard: {detected_text}")
        if keyword in detected_text:
            print("Keyword detected!")
            return True
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print(f"Speech Recognition service error: {e}")
    return False


# Function to handle active listening
def active_listen(recognizer, mic):
    print("Active Listening for user query...")
    try:
        audio = recognizer.listen(mic, timeout=None, phrase_time_limit=15)
        user_query = recognizer.recognize_google(audio)
        print(f"Active Listening Heard Query: {user_query}")
        return user_query
    except sr.UnknownValueError:
        speak_text("Sorry, I couldn't understand that. Please try again.")
    except sr.RequestError as e:
        speak_text("There seems to be an issue with the speech recognition service.")
    return None


def continuous_voice_command():
    print("Listening continuously for voice commands...")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    last_processed_time = 0  # To implement a cooldown
    cooldown = 5  # Cooldown period in seconds

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        while True:
            try:
                print("Listening for a command...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
                user_query = recognizer.recognize_google(audio).lower()
                print(f"Command heard: {user_query}")

                # Safeguards
                if "jarvis" in user_query:  # Soft keyword detection
                    if time.time() - last_processed_time >= cooldown:  # Cooldown check
                        handle_query(user_query.replace("jarvis", "").strip())
                        last_processed_time = time.time()
                    else:
                        print("Cooldown active, ignoring input.")
                else:
                    print("No activation keyword detected, ignoring input.")
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that. Please try again.")
            except sr.RequestError as e:
                print(f"Error with speech recognition: {e}")
                speak_text("There seems to be an issue with speech recognition services.")
            except KeyboardInterrupt:
                print("Exiting...")
                break

def set_volume_from_user(level):
    """Set the system volume to a specific level provided by the user."""
    try:
        level = float(level) / 100.0  # Convert percentage to scalar
        if 0.0 <= level <= 1.0:
            volume.SetMasterVolumeLevelScalar(level, None)
            print(f"Volume set to {int(level * 100)}%")
            speak_text(f"Volume set to {int(level * 100)} percent.")
        else:
            print("Volume level must be between 0 and 100 percent.")
            speak_text("Volume level must be between 0 and 100 percent.")
    except ValueError:
        print("Invalid volume level provided.")
        speak_text("Please provide a valid volume level between 0 and 100.")


def mute_volume():
    """Mute or unmute system volume."""
    is_muted = volume.GetMute()
    if is_muted:
        volume.SetMute(0, None)
        print("Volume unmuted.")
        speak_text("Volume unmuted.")
    else:
        volume.SetMute(1, None)
        print("Volume muted.")
        speak_text("Volume muted.")


def play_song(song_path):
    # Initialize the mixer module of pygame
    pygame.mixer.init()

    # Load the song
    pygame.mixer.music.load(song_path)

    # Play the song
    pygame.mixer.music.play()

    # Wait until the song finishes
    while pygame.mixer.music.get_busy():
        time.sleep(1)

    # Stop the song (optional, as it will stop automatically when done)
    pygame.mixer.music.stop()

def get_time_in_words():


        # Function to convert numbers to words
        def number_to_words(n):
            ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                     "nineteen"]
            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

            if n < 10:
                return ones[n]
            elif 10 <= n < 20:
                return teens[n - 10]
            else:
                ten = n // 10
                one = n % 10
                return tens[ten] + (" " + ones[one] if one != 0 else "")

        # Get the current time
        current_time = datetime.now()
        hour = current_time.hour
        minute = current_time.minute

        # Convert hour and minute to words
        if hour > 12:
            hour -= 12
            period = "PM"
        else:
            period = "AM"

        hour_in_words = number_to_words(hour)
        minute_in_words = number_to_words(minute) if minute > 0 else "o'clock"

        # Return the formatted time in words
        return f"The current time is {hour_in_words} {minute_in_words} {period}"

import psutil
import platform
import shutil
import GPUtil
import os

def system_diagnostics():
    diagnostics = {}

    # OS and system details
    diagnostics["OS"] = platform.system()
    diagnostics["OS Version"] = platform.version()
    diagnostics["Architecture"] = platform.architecture()[0]
    diagnostics["Processor"] = platform.processor()

    # CPU details
    diagnostics["CPU Cores"] = psutil.cpu_count(logical=False)
    diagnostics["Logical Processors"] = psutil.cpu_count(logical=True)
    diagnostics["CPU Usage (%)"] = psutil.cpu_percent(interval=1)

    # Memory details
    memory = psutil.virtual_memory()
    diagnostics["Total RAM (GB)"] = round(memory.total / (1024 ** 3), 2)
    diagnostics["Available RAM (GB)"] = round(memory.available / (1024 ** 3), 2)
    diagnostics["RAM Usage (%)"] = memory.percent

    # Disk details
    disk = shutil.disk_usage("/")
    diagnostics["Total Disk Space (GB)"] = round(disk.total / (1024 ** 3), 2)
    diagnostics["Used Disk Space (GB)"] = round(disk.used / (1024 ** 3), 2)
    diagnostics["Free Disk Space (GB)"] = round(disk.free / (1024 ** 3), 2)
    diagnostics["Disk Usage (%)"] = round((disk.used / disk.total) * 100, 2)

    # GPU details (if available)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({
                "Name": gpu.name,
                "Memory Total (GB)": round(gpu.memoryTotal / 1024, 2),
                "Memory Used (GB)": round(gpu.memoryUsed / 1024, 2),
                "Memory Free (GB)": round(gpu.memoryFree / 1024, 2),
                "GPU Load (%)": gpu.load * 100,
                "Temperature (°C)": gpu.temperature,
            })
        diagnostics["GPU Details"] = gpu_info
    else:
        diagnostics["GPU Details"] = "No GPU detected"

    # Network details
    try:
        net_io = psutil.net_io_counters()
        diagnostics["Bytes Sent (MB)"] = round(net_io.bytes_sent / (1024 ** 2), 2)
        diagnostics["Bytes Received (MB)"] = round(net_io.bytes_recv / (1024 ** 2), 2)
    except Exception as e:
        diagnostics["Network Status"] = f"Error retrieving network stats: {e}"

    return diagnostics



import nmap


def network_scan(target="192.168.1.1/24"):
    scanner = nmap.PortScanner()

    print(f"Scanning network: {target} ...")
    scanner.scan(hosts=target, arguments="-sn")  # Ping scan to find live hosts

    live_hosts = [host for host in scanner.all_hosts() if scanner[host].state() == "up"]

    if not live_hosts:
        return "No active devices found on the network."

    results = {}

    for host in live_hosts:
        print(f"Scanning open ports on {host}...")
        scanner.scan(host, arguments="-p 1-65535 -T4 -sS")  # Full port scan

        open_ports = []
        for proto in scanner[host].all_protocols():
            ports = scanner[host][proto].keys()
            open_ports.extend(ports)

        results[host] = {
            "Open Ports": open_ports if open_ports else "No open ports detected",
            "OS Guess": scanner[host]["osmatch"][0]["name"] if "osmatch" in scanner[host] and scanner[host][
                "osmatch"] else "Unknown",
        }

    return results

# Main Function
if __name__ == "__main__":
    print("Starting J.A.R.V.I.S")
    try:

        continuous_voice_command()
    except KeyboardInterrupt:
        print("Exiting...")