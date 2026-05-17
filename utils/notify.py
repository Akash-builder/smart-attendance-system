import os
import sys
from twilio.rest import Client
from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER

# Add parent dir to path to find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def send_sms(name, current_time):
    """Function to send a quick SMS when attendance is marked"""
    try:
        # Initialize the Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # Construct the message content
        sms_body = f"Attendance Success: {name} was recognized at {current_time}."
        
        # Send message
        message = client.messages.create(
            body=sms_body,
            from_=TWILIO_FROM_NUMBER,
            to=TWILIO_TO_NUMBER
        )
        
        print(f"SMS successfully sent to {TWILIO_TO_NUMBER}. SID: {message.sid}")
        return message.sid
        
    except Exception as e:
        print(f"Failed to send SMS: {e}")
        return None

def send_bulk_sms(student_names, target_phone):
    """Sends a warning message for multiple students with low attendance"""
    results = []
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        for name in student_names:
            try:
                # Warning message template
                alert_text = f"Warning: {name} has low attendance records. Please check."
                
                # Send to the teacher's phone number provided
                msg = client.messages.create(
                    body=alert_text,
                    from_=TWILIO_FROM_NUMBER,
                    to=target_phone
                )
                
                print(f"Alert sent for {name} to {target_phone}")
                results.append((name, msg.sid))
                
            except Exception as student_error:
                print(f"Error sending for {name}: {student_error}")
                results.append((name, None))
                
    except Exception as client_error:
        print(f"Twilio connection error: {client_error}")
        
    return results