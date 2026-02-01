import streamlit as st
import numpy as np
import cv2
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub

# ================= 1. CONFIG STREAMLIT =================
st.set_page_config(page_title="Analyseur Postural Pro (MoveNet)", layout="wide")

# ================= 2. CHARGEMENT MOVENET =================
@st.cache_resource
def load_movenet():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return model

movenet = load_movenet()

# ================= 3. OUTILS =================
def preprocess(image):
    image = tf.image.resize_with_pad(image, 192, 192)
    image = tf.expand_dims(image, axis=0)
    return tf.cast(image, dtype=tf.int32)


def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    dot = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag == 0:
        return 0
    return np.degrees(np.arccos(np.clip(dot / mag, -1, 1)))


def tibia_vertical_angle(knee, ankle):
    v = np.array([ankle[0]-knee[0], ankle[1]-knee[1]])
    vertical = np.array([0, 1])
    dot = np.dot(v, vertical)
    mag = np.linalg.norm(v)
    if mag == 0:
        return 0
    return np.degrees(np.arccos(np.clip(dot / mag, -1, 1)))


def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Bilan Postural", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Nom : {data['Nom']}", ln=True)
    pdf.cell(0, 8, f"Date : {datetime.now().strftime('%d/%m/%Y')}", ln=True)
    pdf.ln(5)
    for k, v in data.items():
        if k not in ['Nom']:
            pdf.cell(0, 8, f"{k} : {v}", ln=True)
    filename = f"Bilan_{data['Nom'].replace(' ', '_')}.pdf"
    pdf.output(filename)
    return filename

# ================= 4. INTERFACE =================
st.title("🧍 Analyseur Postural Pro — MoveNet")

with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille_cm = st.number_input("Taille (cm)", min_value=100, max_value=220, value=170)
    st.divider()
    source = st.radio("Source", ["📷 Caméra", "📁 Photo"])

col_input, col_result = st.columns(2)

with col_input:
    image_data = None
    if source == "📷 Caméra":
        cam_file = st.camera_input("Prendre une photo")
        if cam_file:
            image_data = cam_file
    else:
        upload_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
        if upload_file:
            image_data = upload_file

# ================= 5. ANALYSE =================
if image_data:
    img = Image.open(image_data).convert('RGB')
    img_np = np.array(img)

    # Correction orientation : si paysage → rotation
    if img_np.shape[1] > img_np.shape[0]:
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)

    h, w, _ = img_np.shape

    if st.button("⚙️ LANCER L'ANALYSE", use_container_width=True):
        with st.spinner("Analyse IA en cours..."):
            outputs = movenet.signatures['serving_default'](preprocess(img_np))
            kps = outputs['output_0'][0][0].numpy()

            def pt(i):
                y, x, s = kps[i]
                return np.array([x*w, y*h])

            LS, RS = pt(5), pt(6)
            LH, RH = pt(11), pt(12)
            LK, RK = pt(13), pt(14)
            LA, RA = pt(15), pt(16)

            # Angles
            shoulder_angle = math.degrees(math.atan2(LS[1]-RS[1], LS[0]-RS[0]))
            hip_angle = math.degrees(math.atan2(LH[1]-RH[1], LH[0]-RH[0]))
            knee_l = calculate_angle(LH, LK, LA)
            knee_r = calculate_angle(RH, RK, RA)
            ankle_l = tibia_vertical_angle(LK, LA)
            ankle_r = tibia_vertical_angle(RK, RA)

            # Différences en mm (échelle via taille)
            px_height = max(LA[1], RA[1]) - min(LS[1], RS[1])
            mm_per_px = (taille_cm * 10) / px_height if px_height > 0 else 0
            diff_shoulders_mm = abs(LS[1]-RS[1]) * mm_per_px
            diff_hips_mm = abs(LH[1]-RH[1]) * mm_per_px

            annotated = img_np.copy()
            for p in [LS, RS, LH, RH, LK, RK, LA, RA]:
                cv2.circle(annotated, tuple(p.astype(int)), 5, (0,255,0), -1)

            results = {
                "Nom": nom,
                "Inclinaison épaules": f"{shoulder_angle:.1f}°",
                "Inclinaison bassin": f"{hip_angle:.1f}°",
                "Différence épaules": f"{diff_shoulders_mm:.1f} mm",
                "Différence bassin": f"{diff_hips_mm:.1f} mm",
                "Angle genou gauche": f"{knee_l:.1f}°",
                "Angle genou droit": f"{knee_r:.1f}°",
                "Angle cheville gauche": f"{ankle_l:.1f}°",
                "Angle cheville droite": f"{ankle_r:.1f}°"
            }

            with col_result:
                st.image(annotated, caption="Analyse posturale")
                st.table(results)

                pdf_path = generate_pdf(results)
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Télécharger le PDF", f, file_name=pdf_path)
