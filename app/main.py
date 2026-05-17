import cv2
import datetime
import sys
import os

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.face_loader import load_faces
from utils.recognition import detect_faces, recognize
from utils.notify import send_sms
from database.db import insert_attendance

def main():
    # Load all registered student faces
    print("Initializing student face database...")
    known_vectors, student_names = load_faces()

    if not known_vectors.size:
        print("Warning: No faces found in data/known_faces directory.")

    # Start webcam capture
    video_capture = cv2.VideoCapture(0)
    
    # Track marked students for the current session to avoid spamming the database
    # (Simplified session tracking)
    session_marked = {}

    print("Camera active. Press 'q' or 'ESC' to close.")

    while True:
        success, frame = video_capture.read()
        if not success:
            print("Failed to capture image")
            break

        # Detect faces in the current frame
        face_locations = detect_faces(frame)

        for (x1, y1, x2, y2) in face_locations:
            # Crop the face for recognition
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            # Convert BGR to RGB for the recognition model
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Identify the person
            person_name = recognize(face_rgb, known_vectors, student_names, distance_threshold=1.1)

            # If person is identified, check if we should mark attendance
            if person_name != "Unknown":
                now = datetime.datetime.now()
                timestamp = now.strftime("%H:%M:%S")
                date_today = str(now.date())

                # Check if already marked in this session (e.g., within 1 minute)
                last_time = session_marked.get(person_name)
                if not last_time or (now - last_time).seconds > 60:
                    insert_attendance(person_name, date_today, timestamp)
                    send_sms(person_name, timestamp)
                    session_marked[person_name] = now
                    print(f"Success: Marked attendance for {person_name}")

            # Draw UI on the frame
            color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, person_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show the video feed
        cv2.imshow('Smart Attendance - Camera Feed', frame)

        # Handle keyboard interrupts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()