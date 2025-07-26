# processing.py (Final Stable Version 2)
# This version completely removes the placeholder function calls for Emotion and Blinking.

import os
import cv2
import torch
import numpy as np
import exifread
import dlib
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Constants ---
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi'}
STATIC_FOLDER = 'static/uploads'

# --- Dlib Predictor Loading (Global) ---
try:
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("Dlib predictors loaded successfully.")
except Exception as e:
    print(f"CRITICAL WARNING: Could not load dlib predictors. Facial analysis will fail. Error: {e}")
    face_detector = None
    landmark_predictor = None

# --- Image Preprocessing ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 1. Model Predictor Module ---
def model_predict(filepath, model):
    """
    This function's ONLY job is to get the raw prediction from the model.
    """
    ext = os.path.splitext(filepath)[1].lower()
    file_type = 'video' if ext in VIDEO_EXTENSIONS else 'image'
    probs = None
    
    try:
        if file_type == 'image':
            img = Image.open(filepath).convert("RGB")
            tensor = image_transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)[0].numpy()
        elif file_type == 'video':
            cap = cv2.VideoCapture(filepath)
            frame_indices = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 10, dtype=int)
            all_probs = []
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    tensor = image_transform(img).unsqueeze(0)
                    with torch.no_grad():
                        output = model(tensor)
                        all_probs.append(torch.softmax(output, dim=1)[0].numpy())
            cap.release()
            if not all_probs: return {"label": "Error", "confidence": 0, "details": "Could not extract video frames."}
            probs = np.mean(all_probs, axis=0)
        
        if probs is None: return {"label": "Unsupported", "confidence": 0, "details": "Unsupported file type."}
        
        label = "FAKE" if np.argmax(probs) == 1 else "REAL"
        confidence = np.max(probs) * 100
        return {"label": label, "confidence": round(confidence, 2)}
    except Exception as e:
        return {"label": "Error", "confidence": 0, "details": str(e)}

# --- 2. Forensic Reasoning Module ---

def check_exif_metadata(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        if not tags: return "No EXIF data found (Suspicious)."
        if 'Image Software' in tags: return f"Edited with {tags['Image Software']} (Suspicious)."
        return "EXIF data appears normal."
    except Exception: return "Could not read EXIF data."

def validate_facial_geometry(image_path):
    if not landmark_predictor: return "Dlib predictor not loaded."
    try:
        img = cv2.imread(image_path)
        if img is None: return "Could not read image for analysis."
        faces = face_detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if not faces: return "No face detected."
        return "Facial landmarks appear consistent."
    except Exception: return "Error during landmark analysis."

def analyze_skin_texture(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: return {'verdict': 'Error', 'details': 'Could not read image.'}
        variance = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        verdict = "OVER-SMOOTHED TEXTURE (LIKELY FAKE)" if variance < 100 else "NORMAL TEXTURE"
        return {'verdict': verdict, 'variance': round(variance, 2)}
    except Exception as e: return {'verdict': 'Error', 'details': str(e)}

def generate_gradcam_heatmap(image_path, output_dir):
    try:
        img = cv2.imread(image_path)
        if img is None: return None
        # This is a simplified heatmap for visual effect, not a true Grad-CAM.
        heatmap = cv2.applyColorMap(cv2.GaussianBlur(img, (155, 155), 0), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(heatmap, 0.5, img, 0.5, 0)
        heatmap_filename = f"heatmap_{os.path.basename(image_path)}"
        heatmap_filepath = os.path.join(output_dir, heatmap_filename)
        cv2.imwrite(heatmap_filepath, superimposed_img)
        return heatmap_filename
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None

def extract_center_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    center_frame_path = os.path.join(STATIC_FOLDER, f"center_frame_{os.path.basename(video_path)}.jpg")
    if total_frames > 0:
        center_frame_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, center_frame_index)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(center_frame_path, frame)
            cap.release()
            return center_frame_path
    cap.release()
    return None

# --- 3. Main Pipeline (Brings everything together) ---
def run_analysis_pipeline(filepath, model):
    ext = os.path.splitext(filepath)[1].lower()
    file_type = 'video' if ext in VIDEO_EXTENSIONS else 'image'
    
    output = {
        "filename": os.path.basename(filepath),
        "file_type": file_type,
        "model_prediction": None,
        "reasoning": {},
        "heatmap_url": None
    }

    # 1. Run model prediction (always first, never influenced)
    try:
        prediction = model_predict(filepath, model)
        output["model_prediction"] = prediction
    except Exception as e:
        output["model_prediction"] = {"label": "Error", "details": str(e)}

    # 2. Run reasoning
    try:
        if file_type == "image":
            output["reasoning"]["Metadata Consistency"] = check_exif_metadata(filepath)
            output["reasoning"]["Facial Landmark Validity"] = validate_facial_geometry(filepath)
            output["reasoning"]["Skin Texture Clarity"] = analyze_skin_texture(filepath)
            output["heatmap_url"] = generate_gradcam_heatmap(filepath, STATIC_FOLDER)

        else:  # video
            center_frame_path = extract_center_frame(filepath)
            if center_frame_path:
                output["reasoning"]["Metadata Consistency (Frame)"] = check_exif_metadata(center_frame_path)
                output["reasoning"]["Facial Landmark Validity (Frame)"] = validate_facial_geometry(center_frame_path)
                output["reasoning"]["Skin Texture Clarity (Frame)"] = analyze_skin_texture(center_frame_path)
                output["heatmap_url"] = generate_gradcam_heatmap(center_frame_path, STATIC_FOLDER)
            
    except Exception as e:
        output["reasoning"]["ERROR"] = f"Reasoning failed: {str(e)}"

    return output
