import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image

# Initialize the FaceNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Standard preprocessing for FaceNet
face_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def get_face_vector(image_file):
    """Convert an image file path to a face embedding vector"""
    frame = cv2.imread(image_file)
    if frame is None:
        return None

    # Convert to RGB and then to PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Transform to tensor
    tensor = face_transform(pil_img).unsqueeze(0).to(device)

    # Get the embedding
    with torch.no_grad():
        vector = resnet_model(tensor)

    return vector.cpu().numpy()[0]

def load_faces(folder_path="data/known_faces"):
    """
    Scans the folder for registered face images.
    Supports both subfolders (one per person) or direct files.
    """
    vectors = []
    names = []

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found")
        return np.array([]), []

    # Iterate through the main directory
    for item in os.listdir(folder_path):
        item_full_path = os.path.join(folder_path, item)

        # Case 1: Folder per person (Recommended)
        if os.path.isdir(item_full_path):
            person_name = item
            for img_file in os.listdir(item_full_path):
                if os.path.splitext(img_file)[1].lower() in IMAGE_EXT:
                    img_full_path = os.path.join(item_full_path, img_file)
                    v = get_face_vector(img_full_path)
                    if v is not None:
                        vectors.append(v)
                        names.append(person_name)

        # Case 2: Individual image files directly in the root
        else:
            if os.path.splitext(item)[1].lower() in IMAGE_EXT:
                v = get_face_vector(item_full_path)
                if v is not None:
                    vectors.append(v)
                    names.append(item) # Label is the filename

    if not vectors:
        print("No images found in database.")
        return np.array([]), []

    print(f"Loaded {len(names)} total images for {len(set(names))} students.")
    return np.array(vectors), names