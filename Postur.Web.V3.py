import streamlit as st
import numpy as np
import cv2
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime
import tensorflow as tf
import os

# ================= 1. CONFIG STREAMLIT =================
st.set_page_config(page_title="Analyseur Postural Pro (MoveNet)", layout="wide")

# ================= 2. CHARGEMENT DU MODÈLE MOVENET =================
@st.cache_resource
def load_movenet():
    model = tf.saved_model.load(
        "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        tags=["serve"]
    )
    return model.signatures['serving_default']

movenet = load_movenet()

# ================= 3. FONCTIONS UTILES =================
def preprocess(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize_with_pad(image, 192, 192)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.int32)
    return image

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    dot = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag == 0:
        return 0
    angle = np.arccos(np.clip(dot / mag, -1.0, 1.0))
    return np.degrees(angle)

KEYPOINTS = {
    5: "left_shoulder",
    6: "right_shoulder",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

# ================= 4. INTERFACE =================
st.title("🧍 Analyseur Postural Pro — MoveNet")

with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    st.divider()
    source = st.radio("Source", ["📷 Caméra", "📁 Photo"])

col_input, col_result = st.columns(2)

with col_input:
    image_data = None
    if source == "📷 Caméra":n        cam_file = st.camera_input("Prendre une photo")
        if cam_file:
            image_data = cam_file
    else:
        upload_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
        if upload_file:
            image_data = upload_file

# ================= 5. ANALYSE =================
if image_data:
    img_pil = Image.open(image_data).convert('RGB')
    img_np = np.array(img_pil)
    h, w, _ = img_np.shape

    if st.button("⚙️ LANCER L'ANALYSE", use_container_width=True):
        with st.spinner("Analyse IA en cours..."):
            input_tensor = preprocess(img_np)
            outputs = movenet(input_tensor)
            keypoints = outputs['output_0'][0][0].numpy()

            pts = {}
            for idx, name in KEYPOINTS.items():
                y, x, score = keypoints[idx]
                pts[name] = (int(x * w), int(y * h), score)

            # Angles
            shoulder_angle = math.degrees(math.atan2(
                pts['left_shoulder'][1] - pts['right_shoulder'][1],
                pts['left_shoulder'][0] - pts['right_shoulder'][0]
            ))
            hip_angle = math.degrees(math.atan2(
                pts['left_hip'][1] - pts['right_hip'][1],
                pts['left_hip'][0] - pts['right_hip'][0]
            ))
            knee_l = calculate_angle(
                pts['left_hip'], pts['left_knee'], pts['left_ankle']
            )
            knee_r = calculate_angle(
                pts['right_hip'], pts['right_knee'], pts['right_ankle']
            )

            # Dessin
            annotated = img_np.copy()
            for p in pts.values():
                cv2.circle(annotated, (p[0], p[1]), 5, (0, 255, 0), -1)

            with col_result:
                st.image(annotated, caption="Analyse Posturale (MoveNet)")
                res = {
                    "Nom": nom,
                    "Inclinaison Épaules": f"{shoulder_angle:.1f}°",
                    "Inclinaison Bassin": f"{hip_angle:.1f}°",
                    "Angle Genou Gauche": f"{knee_l:.1f}°",
                    "Angle Genou Droit": f"{knee_r:.1f}°"
                }
                st.write("### 📊 Résultats")
                st.table(res)
