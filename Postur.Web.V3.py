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
def rotate_if_landscape(img):
    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def ensure_uint8_rgb(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mag == 0:
        return 0.0
    return math.degrees(np.arccos(np.clip(np.dot(v1, v2) / mag, -1, 1)))

def pdf_safe(text):
    return (
        str(text)
        .replace("°", " deg")
        .replace("–", "-")
        .replace("—", "-")
        .encode("latin-1", errors="ignore")
        .decode("latin-1")
    )

# =========================
# PDF (COMPAT TOUTES VERSIONS FPDF)
# =========================
def generate_pdf(data, img_rgb):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "BILAN POSTURAL IA", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Patient : {pdf_safe(data.get('Nom',''))}", ln=True)
    pdf.cell(0, 8, datetime.now().strftime("%d/%m/%Y %H:%M"), ln=True)
    pdf.ln(5)

    tmp = os.path.join(tempfile.gettempdir(), "img.png")
    Image.fromarray(img_rgb).save(tmp)
    pdf.image(tmp, x=40, w=130)
    os.remove(tmp)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(120, 8, "Indicateur", 1)
    pdf.cell(60, 8, "Valeur", 1, ln=True)

    pdf.set_font("Arial", "", 11)
    for k, v in data.items():
        if k != "Nom":
            pdf.cell(120, 8, pdf_safe(k), 1)
            pdf.cell(60, 8, pdf_safe(v), 1, ln=True)

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1")

# =========================
# SESSION
# =========================
if "override_one" not in st.session_state:
    st.session_state.override_one = {}

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    nom = st.text_input("Nom", "Anonyme")
    taille_cm = st.number_input("Taille (cm)", 100, 220, 170)

    source = st.radio("Source image", ["📷 Caméra", "📁 Fichier"])
    enable_edit = st.checkbox("Correction par clic", True)

    editable_points = [
        "Hanche G", "Hanche D",
        "Genou G", "Genou D",
        "Cheville G", "Cheville D",
        "Talon G", "Talon D"
    ]
    point_to_edit = st.selectbox("Point à corriger", editable_points)

    if st.button("🧹 Reset corrections"):
        st.session_state.override_one = {}

col_input, col_result = st.columns(2)

# =========================
# IMAGE INPUT
# =========================
with col_input:
    image_data = st.camera_input("Capture") if source == "📷 Caméra" else st.file_uploader("Image", ["jpg", "png"])

if not image_data:
    st.stop()

img = np.array(Image.open(image_data).convert("RGB"))
img = ensure_uint8_rgb(rotate_if_landscape(img))
h, w = img.shape[:2]

# =========================
# PREVIEW CLIQUABLE
# =========================
with col_input:
    res = pose.process(img)
    if not res.pose_landmarks:
        st.warning("Pose non détectée")
        st.stop()

    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(e): return np.array([lm[e.value].x * w, lm[e.value].y * h])

    points = {
        "Hanche G": pt(L.LEFT_HIP),
        "Hanche D": pt(L.RIGHT_HIP),
        "Genou G": pt(L.LEFT_KNEE),
        "Genou D": pt(L.RIGHT_KNEE),
        "Cheville G": pt(L.LEFT_ANKLE),
        "Cheville D": pt(L.RIGHT_ANKLE),
        "Talon G": pt(L.LEFT_HEEL),
        "Talon D": pt(L.RIGHT_HEEL),
    }

    disp = img.copy()
    for p in points.values():
        cv2.circle(disp, tuple(p.astype(int)), 6, (0,255,0), -1)

    coords = streamlit_image_coordinates(Image.fromarray(disp))
    if enable_edit and coords:
        st.session_state.override_one[point_to_edit] = np.array([coords["x"], coords["y"]])

    st.image(disp, use_column_width=True)

# =========================
# ANALYSE
# =========================
with col_result:
    if not st.button("▶ Lancer l'analyse"):
        st.stop()

    for k, v in st.session_state.override_one.items():
        if k in points:
            points[k] = v

    LS = pt(L.LEFT_SHOULDER)
    RS = pt(L.RIGHT_SHOULDER)

    shoulder_angle = abs(math.degrees(math.atan2(LS[1]-RS[1], LS[0]-RS[0])))

    ann = img.copy()
    for p in points.values():
        cv2.circle(ann, tuple(p.astype(int)), 7, (0,255,0), -1)

    cv2.line(ann, tuple(LS.astype(int)), tuple(RS.astype(int)), (255,0,0), 3)

    results = {
        "Nom": nom,
        "Inclinaison épaules": f"{shoulder_angle:.1f}°"
    }

    st.subheader("Résultats")
    st.table(results)

    st.subheader("Image annotée")
    st.image(ann, use_column_width=True)

    st.subheader("PDF")
    pdf_bytes = generate_pdf(results, ann)
    st.download_button(
        "📥 Télécharger le PDF",
        pdf_bytes,
        f"Bilan_{nom.replace(' ','_')}.pdf",
        "application/pdf"
    )
