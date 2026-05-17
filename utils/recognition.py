import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

# Load the Haar Cascade for face detection
# This is a pre-trained model provided by OpenCV
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Setup FaceNet model using facenet-pytorch
# We use InceptionResnetV1 which is very accurate for face recognition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Image transformation: Resize to 160x160 and convert to tensor
# The model expects normalized input in range [-1, 1]
face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def detect_faces(img):
    """Detect all faces in the given BGR frame"""
    # Convert to grayscale for detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Run the detector
    boxes = face_detector.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    
    # Return boxes as (x1, y1, x2, y2)
    coords = []
    for (x, y, w, h) in boxes:
        coords.append((int(x), int(y), int(x + w), int(y + h)))
    return coords

def extract_embedding(face_image):
    """Generate the unique 512-dimension vector for a face"""
    if isinstance(face_image, np.ndarray):
        face_image = Image.fromarray(face_image)

    # Transform and add batch dimension
    img_tensor = face_transform(face_image).unsqueeze(0).to(device)

    # Pass through the neural network
    with torch.no_grad():
        features = resnet_model(img_tensor)

    return features.cpu().numpy()[0]

def recognize(face_crop, db_embeddings, db_labels, distance_threshold=1.1):
    """Compare detected face against the database of known students"""
    if face_crop is None or face_crop.size == 0:
        return "Unknown"

    if len(db_embeddings) == 0:
        return "Unknown"

    # Get the embedding for the current face
    current_emb = extract_embedding(face_crop)

    # Calculate distance to every known face in the database
    # Lower distance means the faces are more similar
    diffs = np.linalg.norm(db_embeddings - current_emb, axis=1)
    best_score = float(np.min(diffs))
    best_index = int(np.argmin(diffs))

    # For debugging/tuning
    # print(f"Recognition score: {best_score}")

    if best_score < distance_threshold:
        return db_labels[best_index]

    return "Unknown"