import streamlit as st
st.set_page_config(page_title="Analyseur Postural Pro (MediaPipe)", layout="wide")

import os
import tempfile
import numpy as np
import cv2
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime
import io

import mediapipe as mp
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("🧍 Analyseur Postural Pro (MediaPipe)")
st.markdown("---")

# =========================
# 1) MEDIAPIPE
# =========================
mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

pose = load_pose()

# =========================
# 2) OUTILS
# =========================
def rotate_if_landscape(img_np_rgb):
    if img_np_rgb.shape[1] > img_np_rgb.shape[0]:
        img_np_rgb = cv2.rotate(img_np_rgb, cv2.ROTATE_90_CLOCKWISE)
    return img_np_rgb

def ensure_uint8_rgb(img):
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img *= 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)

def to_png_bytes(img):
    img = ensure_uint8_rgb(img)
    pil = Image.fromarray(img, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(dot / mag, -1, 1)))

def pdf_safe(text):
    if text is None:
        return ""
    return (
        str(text)
        .replace("°", " deg")
        .replace("–", "-")
        .replace("—", "-")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .encode("latin-1", errors="ignore")
        .decode("latin-1")
    )

def generate_pdf(data, img):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(31, 73, 125)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(0, 20, "BILAN POSTURAL IA", ln=True, align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(25)
    pdf.cell(110, 10, f"Patient : {pdf_safe(data.get('Nom',''))}")
    pdf.set_font("Arial", '', 11)
    pdf.cell(80, 10, datetime.now().strftime('%d/%m/%Y %H:%M'), ln=1, align="R")

    tmp = os.path.join(tempfile.gettempdir(), "posture.png")
    Image.fromarray(img).save(tmp)
    pdf.image(tmp, x=55, w=100)
    os.remove(tmp)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(120, 10, "Indicateur", 1, 0, 'L', True)
    pdf.cell(70, 10, "Valeur", 1, 1, 'C', True)

    pdf.set_font("Arial", '', 11)
    for k, v in data.items():
        if k != "Nom":
            pdf.cell(120, 9, pdf_safe(k), 1)
            pdf.cell(70, 9, pdf_safe(v), 1, ln=1, align="C")

    return pdf.output(dest="S").encode("latin-1")

# =========================
# 3) SESSION
# =========================
if "override_one" not in st.session_state:
    st.session_state.override_one = {}

# =========================
# 4) UI
# =========================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom complet", "Anonyme")
    taille_cm = st.number_input("Taille (cm)", 100, 220, 170)

    source = st.radio("Source de l'image", ["📷 Caméra", "📁 Téléverser une photo"])
    enable_click_edit = st.checkbox("Activer correction par clic", True)

    editable_points = [
        "Hanche G", "Hanche D",
        "Genou G", "Genou D",
        "Cheville G", "Cheville D",
        "Talon G", "Talon D",
    ]
    point_to_edit = st.selectbox("Point à corriger", editable_points)

    if st.button("🧹 Reset tout"):
        st.session_state.override_one = {}

col_input, col_result = st.columns(2)

# =========================
# 5) IMAGE
# =========================
with col_input:
    image_data = st.camera_input("Capture") if source == "📷 Caméra" else st.file_uploader("Image", ["jpg", "png", "jpeg"])
if not image_data:
    st.stop()

img = np.array(Image.open(image_data).convert("RGB"))
img = ensure_uint8_rgb(rotate_if_landscape(img))
h, w = img.shape[:2]

# =========================
# 6) PREVIEW
# =========================
with col_input:
    disp_w = min(900, w)
    scale = disp_w / w
    img_disp = cv2.resize(img, (disp_w, int(h * scale)))

    res = pose.process(img)
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        L = mp_pose.PoseLandmark
        def pt(e): return np.array([lm[e.value].x * w, lm[e.value].y * h])

        base = {
            "Hanche G": pt(L.LEFT_HIP),
            "Hanche D": pt(L.RIGHT_HIP),
            "Genou G": pt(L.LEFT_KNEE),
            "Genou D": pt(L.RIGHT_KNEE),
            "Cheville G": pt(L.LEFT_ANKLE),
            "Cheville D": pt(L.RIGHT_ANKLE),
            "Talon G": pt(L.LEFT_HEEL),
            "Talon D": pt(L.RIGHT_HEEL),
        }

        preview = img_disp.copy()
        for k, p in base.items():
            x, y = int(p[0]*scale), int(p[1]*scale)
            cv2.circle(preview, (x,y), 6, (0,255,0), -1)

        coords = streamlit_image_coordinates(Image.fromarray(preview))
        if enable_click_edit and coords:
            st.session_state.override_one[point_to_edit] = (coords["x"]/scale, coords["y"]/scale)

        st.image(preview, use_container_width=True)

# =========================
# 7) ANALYSE
# =========================
with col_result:
    if not st.button("▶ Lancer l'analyse"):
        st.stop()

# calculs inchangés (repris de ta version)
# ...

# =========================
# 8) SORTIE
# =========================
with col_result:
    st.image(img, caption="Image annotée", use_container_width=True)
    pdf = generate_pdf({"Nom": nom}, img)
    st.download_button("📥 Télécharger le Bilan PDF", pdf, "bilan_postural.pdf", "application/pdf")
