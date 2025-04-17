import cv2
import face_recognition
import os
import time
import csv
import pickle
import json
import base64
import numpy as np
from flask import Flask, jsonify, request, render_template_string, Response
from io import BytesIO
from PIL import Image
import io

app = Flask(__name__)

# Global variables
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
pickle_file = os.path.join(DATA_DIR, "face_encodings.pkl")
attendance_file = os.path.join(DATA_DIR, "attendance.json")
csv_file = os.path.join(DATA_DIR, 'students.csv')

known_face_encodings = []
known_face_ids = []
roll_to_name = {}
attendance = set()
last_unknown_face = None


# Create necessary directories
def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def load_roll_to_name():
    global roll_to_name
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:
                    roll, name = row[0], row[1]
                    roll_to_name[roll] = name
    else:
        # Create empty CSV file if it doesn't exist
        with open(csv_file, 'w', newline='') as file:
            pass


def load_faces_from_directory():
    face_encodings = []
    face_ids = []
    try:
        if os.path.exists(IMAGES_DIR):
            for filename in os.listdir(IMAGES_DIR):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(IMAGES_DIR, filename)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        face_encodings.append(encodings[0])
                        person_id = os.path.splitext(filename)[0]
                        face_ids.append(person_id)
                        print(f"Loaded reference image for ID: {person_id}")
    except Exception as e:
        print(f"Error loading images: {str(e)}")
    return face_encodings, face_ids


def load_known_faces():
    global known_face_encodings, known_face_ids
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            known_face_encodings, known_face_ids = pickle.load(file)
            print("Encodings loaded from file!")
    else:
        print("No saved encodings found. Loading from images...")
        known_face_encodings, known_face_ids = load_faces_from_directory()
        save_known_faces()


def save_known_faces():
    with open(pickle_file, 'wb') as file:
        pickle.dump((known_face_encodings, known_face_ids), file)
    print("Saved face encodings to file.")


def save_attendance():
    attendance_data = {
        'date': time.strftime('%Y-%m-%d'),
        'time': time.strftime('%H:%M:%S'),
        'present': [
            {'roll': roll, 'name': roll_to_name.get(roll, 'Unknown')}
            for roll in sorted(attendance)
        ]
    }
    with open(attendance_file, 'w') as file:
        json.dump(attendance_data, file, indent=4)
    print("Attendance JSON saved to file.")


def process_frame(frame_data):
    global attendance, last_unknown_face
    try:
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        image_bytes = base64.b64decode(frame_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        recognized_faces = []
        unknown_detected = False
        for i, (top, right, bottom, left), face_encoding in zip(range(len(face_locations)), face_locations,
                                                                face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                roll = known_face_ids[first_match_index]
                name = roll_to_name.get(roll, roll)
                attendance.add(roll)
                recognized_faces.append(
                    {"name": name, "roll": roll, "status": "known", "box": [left, top, right, bottom]})
            else:
                unknown_detected = True
                face_img = frame[top:bottom, left:right]
                last_unknown_face = {'face_img': face_img.copy(), 'frame': frame.copy(),
                                     'location': (top, right, bottom, left)}
                recognized_faces.append({"name": "Unknown", "status": "unknown", "box": [left, top, right, bottom]})
        return {"faces": recognized_faces, "unknown_detected": unknown_detected, "attendance_count": len(attendance)}
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return {"error": str(e)}


def add_new_student(face_img, roll, name):
    global known_face_encodings, known_face_ids, roll_to_name, attendance
    if roll in known_face_ids:
        return False, "Roll number already exists"
    try:
        # Save image
        image_path = os.path.join(IMAGES_DIR, f"{roll}.png")
        cv2.imwrite(image_path, face_img)

        # Update CSV
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([roll, name])

        roll_to_name[roll] = name

        # Update face encodings
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            return False, "Failed to encode face"
        known_face_encodings.append(encodings[0])
        known_face_ids.append(roll)
        save_known_faces()
        attendance.add(roll)
        return True, "Student added successfully"
    except Exception as e:
        return False, f"Error: {str(e)}"


@app.route('/')
def index():
    """Render the main application UI"""
    html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Face Recognition Attendance System</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; max-width: 1000px; margin: 0 auto; padding: 20px; }
                .container { display: flex; flex-direction: column; align-items: center; }
                .camera-container { position: relative; width: 640px; max-width: 100%; margin: 20px 0; }
                #video { width: 100%; background-color: #f0f0f0; border-radius: 8px; }
                canvas { display: none; }
                .controls { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin: 15px 0; }
                button { padding: 12px 24px; background-color: #4CAF50; color: white; 
                        border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
                button:hover { background-color: #45a049; }
                button:disabled { background-color: #cccccc; cursor: not-allowed; }
                #stopBtn { background-color: #f44336; }
                #stopBtn:hover { background-color: #d32f2f; }
                #registerBtn { background-color: #2196F3; }
                #registerBtn:hover { background-color: #0b7dda; }

                h1 { color: #2C3E50; text-align: center; }
                .status { margin-top: 20px; padding: 10px; background-color: #f2f2f2; border-radius: 4px; text-align: center; }

                /* Modal styles */
                .modal { display: none; position: fixed; z-index: 1; left: 0; top: 0; width: 100%; height: 100%; 
                       overflow: auto; background-color: rgba(0,0,0,0.4); }
                .modal-content { background-color: #fefefe; margin: 15% auto; padding: 20px; 
                               border: 1px solid #888; width: 80%; max-width: 500px; border-radius: 5px; }
                .close { color: #aaa; float: right; font-size: 28px; font-weight: bold; }
                .close:hover, .close:focus { color: black; text-decoration: none; cursor: pointer; }

                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; }
                input[type="text"] { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
                .submit-btn { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; 
                            border-radius: 4px; cursor: pointer; }
                .error-message { color: #f44336; margin-top: 10px; }

                #faceImage { max-width: 100%; height: auto; margin-bottom: 15px; display: block; }
                .face-box { position: absolute; border: 3px solid; pointer-events: none; }
                .face-label { position: absolute; background: rgba(0,0,0,0.7); color: white; padding: 2px 5px; font-size: 14px; }

                .table-container { width: 100%; max-width: 800px; margin: 20px auto; }
                table { width: 100%; border-collapse: collapse; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }

                @media (max-width: 768px) {
                    .camera-container { width: 100%; }
                    .controls { flex-direction: column; width: 100%; }
                    button { width: 100%; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Face Recognition Attendance System</h1>

                <div class="camera-container">
                    <video id="video" autoplay></video>
                    <canvas id="canvas"></canvas>
                    <!-- Face boxes will be added here dynamically -->
                </div>

                <div class="controls">
                    <button id="startBtn">Start Camera</button>
                    <button id="stopBtn" disabled>Stop Camera</button>
                    <button id="registerBtn" disabled>Register New Student</button>
                    <button id="viewBtn">View Attendance</button>
                    <button id="clearBtn">Clear Attendance</button>
                </div>

                <div class="status">
                    <p id="statusText">Camera is stopped</p>
                    <p id="attendanceCount">Students present: 0</p>
                </div>

                <div class="table-container" id="attendanceTableContainer" style="display: none;">
                    <h2>Current Attendance</h2>
                    <table id="attendanceTable">
                        <thead>
                            <tr>
                                <th>Roll Number</th>
                                <th>Name</th>
                            </tr>
                        </thead>
                        <tbody id="attendanceBody">
                            <!-- Attendance data will be populated here -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Registration Modal -->
            <div id="registerModal" class="modal">
                <div class="modal-content">
                    <span class="close">&times;</span>
                    <h2>Register New Student</h2>

                    <img id="faceImage" src="" alt="Detected Face">

                    <div class="form-group">
                        <label for="rollInput">Roll Number:</label>
                        <input type="text" id="rollInput" placeholder="Enter roll number">
                    </div>

                    <div class="form-group">
                        <label for="nameInput">Student Name:</label>
                        <input type="text" id="nameInput" placeholder="Enter student name">
                    </div>

                    <p id="errorMsg" class="error-message"></p>

                    <button class="submit-btn" id="submitRegistrationBtn">Register Student</button>
                </div>
            </div>

            <script>
                // Global variables
                let video = document.getElementById('video');
                let canvas = document.getElementById('canvas');
                let ctx = canvas.getContext('2d');
                let cameraStream = null;
                let processingInterval = null;
                let unknownFaceDetected = false;

                // Buttons
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
                const registerBtn = document.getElementById('registerBtn');
                const viewBtn = document.getElementById('viewBtn');
                const clearBtn = document.getElementById('clearBtn');

                // Modal elements
                const registerModal = document.getElementById('registerModal');
                const closeModalBtn = document.querySelector('.close');
                const submitRegistrationBtn = document.getElementById('submitRegistrationBtn');

                // Start camera
                startBtn.addEventListener('click', async () => {
                    try {
                        cameraStream = await navigator.mediaDevices.getUserMedia({ 
                            video: { 
                                width: { ideal: 640 },
                                height: { ideal: 480 },
                                facingMode: "user"
                            } 
                        });

                        video.srcObject = cameraStream;
                        canvas.width = video.clientWidth;
                        canvas.height = video.clientHeight;

                        // Wait for video to load
                        video.onloadedmetadata = () => {
                            startBtn.disabled = true;
                            stopBtn.disabled = false;
                            document.getElementById('statusText').textContent = "Camera is running";

                            // Start processing frames
                            processingInterval = setInterval(processVideoFrame, 1000); // Process every 1 second
                        };
                    } catch (error) {
                        console.error('Error accessing camera:', error);
                        alert('Could not access the camera. Please make sure camera permissions are enabled.');
                    }
                });

                // Stop camera
                stopBtn.addEventListener('click', () => {
                    stopCamera();
                    fetchAttendance(); // Update attendance list
                });

                // Register new student
                registerBtn.addEventListener('click', () => {
                    if (!unknownFaceDetected) {
                        alert('No unknown face detected yet.');
                        return;
                    }

                    // Get the latest frame and display it
                    captureCurrentFrame().then(imageData => {
                        document.getElementById('faceImage').src = imageData;
                        document.getElementById('errorMsg').textContent = '';
                        document.getElementById('rollInput').value = '';
                        document.getElementById('nameInput').value = '';
                        registerModal.style.display = 'block';
                    });
                });

                // Submit registration
                submitRegistrationBtn.addEventListener('click', () => {
                    const roll = document.getElementById('rollInput').value.trim();
                    const name = document.getElementById('nameInput').value.trim();

                    if(!roll || !name) {
                        document.getElementById('errorMsg').textContent = 'Roll number and name are required';
                        return;
                    }

                    fetch('/register_student', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ roll: roll, name: name }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        if(data.status === 'success') {
                            alert('Student registered successfully!');
                            registerModal.style.display = 'none';
                            unknownFaceDetected = false;
                            registerBtn.disabled = true;
                        } else {
                            document.getElementById('errorMsg').textContent = data.message;
                        }
                    });
                });

                // View attendance
                viewBtn.addEventListener('click', () => {
                    const tableContainer = document.getElementById('attendanceTableContainer');
                    if (tableContainer.style.display === 'none') {
                        fetchAttendance();
                        tableContainer.style.display = 'block';
                        viewBtn.textContent = 'Hide Attendance';
                    } else {
                        tableContainer.style.display = 'none';
                        viewBtn.textContent = 'View Attendance';
                    }
                });

                // Clear attendance
                clearBtn.addEventListener('click', () => {
                    if (confirm('Are you sure you want to clear the attendance?')) {
                        fetch('/clear_attendance', { method: 'POST' })
                            .then(response => response.json())
                            .then(data => {
                                alert(data.message);
                                document.getElementById('attendanceCount').textContent = 'Students present: 0';
                                document.getElementById('attendanceBody').innerHTML = '';
                            });
                    }
                });

                // Close modal
                closeModalBtn.addEventListener('click', () => {
                    registerModal.style.display = 'none';
                });

                // Close modal when clicking outside
                window.addEventListener('click', (event) => {
                    if (event.target === registerModal) {
                        registerModal.style.display = 'none';
                    }
                });

                // Process video frame and send to server
                function processVideoFrame() {
                    if (!cameraStream) return;

                    // Draw video frame to canvas
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                    // Convert canvas to base64
                    const imageData = canvas.toDataURL('image/jpeg', 0.8);

                    // Send to server for processing
                    fetch('/process_frame', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ image: imageData }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            console.error('Error processing frame:', data.error);
                            return;
                        }

                        // Update UI with recognition results
                        updateFaceBoxes(data.faces);
                        document.getElementById('attendanceCount').textContent = `Students present: ${data.attendance_count}`;

                        // Enable/disable register button
                        unknownFaceDetected = data.unknown_detected;
                        registerBtn.disabled = !unknownFaceDetected;
                    });
                }

                // Update face boxes on video
                function updateFaceBoxes(faces) {
                    // Remove existing face boxes
                    document.querySelectorAll('.face-box, .face-label').forEach(el => el.remove());

                    // Add new face boxes
                    const container = document.querySelector('.camera-container');
                    const videoRect = video.getBoundingClientRect();

                    faces.forEach(face => {
                        const [left, top, right, bottom] = face.box;

                        // Calculate position relative to video size
                        const boxWidth = right - left;
                        const boxHeight = bottom - top;

                        // Calculate scaling factors
                        const scaleX = videoRect.width / canvas.width;
                        const scaleY = videoRect.height / canvas.height;

                        // Create box element
                        const boxElement = document.createElement('div');
                        boxElement.className = 'face-box';
                        boxElement.style.left = `${left * scaleX}px`;
                        boxElement.style.top = `${top * scaleY}px`;
                        boxElement.style.width = `${boxWidth * scaleX}px`;
                        boxElement.style.height = `${boxHeight * scaleY}px`;
                        boxElement.style.borderColor = face.status === 'known' ? '#4CAF50' : '#f44336';

                        // Create label element
                        const labelElement = document.createElement('div');
                        labelElement.className = 'face-label';
                        labelElement.style.left = `${left * scaleX}px`;
                        labelElement.style.top = `${(top * scaleY) - 20}px`;
                        labelElement.textContent = face.status === 'known' ? face.name : 'Unknown';

                        container.appendChild(boxElement);
                        container.appendChild(labelElement);
                    });
                }

                // Capture current frame
                async function captureCurrentFrame() {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    return canvas.toDataURL('image/jpeg', 0.8);
                }

                // Stop camera
                function stopCamera() {
                    if (cameraStream) {
                        cameraStream.getTracks().forEach(track => track.stop());
                        cameraStream = null;
                    }

                    if (processingInterval) {
                        clearInterval(processingInterval);
                        processingInterval = null;
                    }

                    video.srcObject = null;
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    registerBtn.disabled = true;
                    document.getElementById('statusText').textContent = "Camera is stopped";

                    // Remove face boxes
                    document.querySelectorAll('.face-box, .face-label').forEach(el => el.remove());
                }

                // Fetch attendance data
                function fetchAttendance() {
                    fetch('/attendance_data')
                        .then(response => response.json())
                        .then(data => {
                            const tableBody = document.getElementById('attendanceBody');
                            tableBody.innerHTML = '';

                            if (data.attendance.length === 0) {
                                const row = document.createElement('tr');
                                row.innerHTML = '<td colspan="2">No students present</td>';
                                tableBody.appendChild(row);
                            } else {
                                data.attendance.forEach(student => {
                                    const row = document.createElement('tr');
                                    row.innerHTML = `<td>${student.roll}</td><td>${student.name}</td>`;
                                    tableBody.appendChild(row);
                                });
                            }
                        });
                }

                // Handle window resize
                window.addEventListener('resize', () => {
                    if (video.videoWidth) {
                        canvas.width = video.clientWidth;
                        canvas.height = video.clientHeight;
                    }
                });
            </script>
        </body>
        </html>
        """
    return render_template_string(html)


@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data received"}), 400
    result = process_frame(data['image'])
    return jsonify(result)


@app.route('/register_student', methods=['POST'])
def register_student():
    global last_unknown_face
    data = request.json
    if not data or 'roll' not in data or 'name' not in data:
        return jsonify({"status": "error", "message": "Missing required data"})
    if last_unknown_face is None:
        return jsonify({"status": "error", "message": "No unknown face available"})
    roll = data['roll']
    name = data['name']
    success, message = add_new_student(last_unknown_face['face_img'], roll, name)
    if success:
        last_unknown_face = None
        return jsonify({"status": "success", "message": message})
    else:
        return jsonify({"status": "error", "message": message})


@app.route('/attendance_data')
def get_attendance_data():
    students_present = [
        {'roll': roll, 'name': roll_to_name.get(roll, 'Unknown')}
        for roll in sorted(attendance)
    ]
    return jsonify({'attendance': students_present})


@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    global attendance
    attendance = set()
    save_attendance()
    return jsonify({'status': 'success', 'message': 'Attendance cleared successfully'})


def initialize_app():
    ensure_directories()
    load_roll_to_name()
    load_known_faces()
    print("Face recognition system initialized successfully!")


initialize_app()

if __name__ == '__main__':
    app.run(debug=True)