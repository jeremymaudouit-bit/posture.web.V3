import streamlit as st
import numpy as np
import cv2
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub
import os

# ================= 1. CONFIG STREAMLIT =================
st.set_page_config(page_title="Analyseur Postural Pro", layout="wide")

# ================= 2. CHARGEMENT MOVENET =================
@st.cache_resource
def load_movenet():
    # Utilisation du modèle Lightning (rapide)
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    return model

movenet = load_movenet()

# ================= 3. OUTILS TECHNIQUES =================
def preprocess(image):
    image = tf.image.resize_with_pad(image, 192, 192)
    image = tf.expand_dims(image, axis=0)
    return tf.cast(image, dtype=tf.int32)

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    dot = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag == 0: return 0
    return math.degrees(math.acos(np.clip(dot / mag, -1, 1)))

def tibia_vertical_angle(knee, ankle):
    v = np.array([ankle[0]-knee[0], ankle[1]-knee[1]])
    vertical = np.array([0, 1])
    dot = np.dot(v, vertical)
    mag = np.linalg.norm(v)
    if mag == 0: return 0
    return math.degrees(math.acos(np.clip(dot / mag, -1, 1)))

def generate_pdf(data, img_np):
    pdf = FPDF()
    pdf.add_page()
    
    # Design de l'en-tête
    pdf.set_fill_color(31, 73, 125) 
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 20, "BILAN POSTURAL IA", ln=True, align="C")
    
    # Infos Patient
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(25)
    pdf.cell(100, 10, f"Patient : {data['Nom']}", ln=0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(90, 10, f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1, align="R")
    pdf.line(10, 68, 200, 68)
    pdf.ln(5)

    # Sauvegarde et insertion de l'image
    img_pil = Image.fromarray(img_np)
    img_path = "temp_analysis.png"
    img_pil.save(img_path)
    pdf.image(img_path, x=60, w=90)
    pdf.ln(5)

    # Tableau des résultats
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(110, 10, "Indicateur de Mesure", 1, 0, 'L', True)
    pdf.cell(80, 10, "Valeur", 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 11)
    for k, v in data.items():
        if k != "Nom":
            pdf.cell(110, 9, f" {k}", 1, 0, 'L')
            pdf.cell(80, 9, f" {v}", 1, 1, 'C')

    # Footer
    pdf.set_y(-25)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, "Document généré par Analyseur Postural Pro - Usage indicatif uniquement.", align="C")

    filename = f"Bilan_{data['Nom'].replace(' ', '_')}.pdf"
    pdf.output(filename)
    return filename

# ================= 4. INTERFACE UTILISATEUR =================
st.title("🧍 Analyseur Postural Pro")
st.markdown("---")

with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille_cm = st.number_input("Taille (cm)", min_value=100, max_value=220, value=170)
    st.divider()
    source = st.radio("Source de l'image", ["📷 Caméra", "📁 Téléverser une photo"])

col_input, col_result = st.columns([1, 1])

image_data = None
with col_input:
    if source == "📷 Caméra":
        image_data = st.camera_input("Capturez la posture de face")
    else:
        image_data = st.file_uploader("Format JPG/PNG", type=["jpg", "png", "jpeg"])

# ================= 5. COEUR DE L'ANALYSE =================
if image_data:
    img = Image.open(image_data).convert('RGB')
    img_np = np.array(img)

    # Correction d'orientation automatique
    if img_np.shape[1] > img_np.shape[0]:
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)

    h, w, _ = img_np.shape

    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE", use_container_width=True):
        with st.spinner("L'IA détecte les points anatomiques..."):
            # Inférence MoveNet
            input_img = preprocess(img_np)
            outputs = movenet.signatures['serving_default'](input_img)
            kps = outputs['output_0'][0][0].numpy()

            def pt(i):
                y, x, s = kps[i]
                return np.array([x*w, y*h])

            # Points clés
            LS, RS = pt(5), pt(6)   # Épaules
            LH, RH = pt(11), pt(12) # Bassin
            LK, RK = pt(13), pt(14) # Genoux
            LA, RA = pt(15), pt(16) # Chevilles

            # --- Calcul des angles (Référence 0° = Horizontal) ---
            # Épaules
            raw_shoulder_angle = math.degrees(math.atan2(LS[1]-RS[1], LS[0]-RS[0]))
            shoulder_angle = abs(raw_shoulder_angle)
            if shoulder_angle > 90: shoulder_angle = abs(shoulder_angle - 180)

            # Bassin
            raw_hip_angle = math.degrees(math.atan2(LH[1]-RH[1], LH[0]-RH[0]))
            hip_angle = abs(raw_hip_angle)
            if hip_angle > 90: hip_angle = abs(hip_angle - 180)

            # --- Articulations et mm ---
            knee_l = calculate_angle(LH, LK, LA)
            knee_r = calculate_angle(RH, RK, RA)
            ankle_l = tibia_vertical_angle(LK, LA)
            ankle_r = tibia_vertical_angle(RK, RA)

            px_height = max(LA[1], RA[1]) - min(LS[1], RS[1])
            mm_per_px = (taille_cm * 10) / px_height if px_height > 0 else 0
            diff_shoulders_mm = abs(LS[1]-RS[1]) * mm_per_px
            diff_hips_mm = abs(LH[1]-RH[1]) * mm_per_px

            # Annotation visuelle
            annotated = img_np.copy()
            points_list = [LS, RS, LH, RH, LK, RK, LA, RA]
            for p in points_list:
                cv2.circle(annotated, tuple(p.astype(int)), 8, (0, 255, 0), -1)
            
            # Dessin des lignes d'inclinaison
            cv2.line(annotated, tuple(LS.astype(int)), tuple(RS.astype(int)), (255, 0, 0), 3)
            cv2.line(annotated, tuple(LH.astype(int)), tuple(RH.astype(int)), (255, 0, 0), 3)

            results = {
                "Nom": nom,
                "Inclinaison Épaules (Horizon = 0°)": f"{shoulder_angle:.1f}°",
                "Inclinaison Bassin (Horizon = 0°)": f"{hip_angle:.1f}°",
                "Dénivelé Épaules (mm)": f"{diff_shoulders_mm:.1f} mm",
                "Dénivelé Bassin (mm)": f"{diff_hips_mm:.1f} mm",
                "Angle Genou Gauche": f"{knee_l:.1f}°",
                "Angle Genou Droit": f"{knee_r:.1f}°",
                "Inclinaison Tibia G / Verticale": f"{ankle_l:.1f}°",
                "Inclinaison Tibia D / Verticale": f"{ankle_r:.1f}°"
            }

            with col_result:
                st.subheader("Résultats de l'analyse")
                st.image(annotated, use_container_width=True)
                st.table(results)

                pdf_path = generate_pdf(results, annotated)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="📥 Télécharger le Bilan PDF",
                        data=f,
                        file_name=pdf_path,
                        mime="application/pdf",
                        use_container_width=True
                    )
                # Nettoyage fichier temporaire
                if os.path.exists("temp_analysis.png"):
                    os.remove("temp_analysis.png")
