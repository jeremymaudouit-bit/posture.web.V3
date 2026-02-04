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

import mediapipe as mp
from streamlit_image_coordinates import streamlit_image_coordinates  # ✅ NEW

st.title("🧍 Analyseur Postural Pro (MediaPipe)")
st.markdown("---")

# ================= 1. CHARGEMENT MEDIAPIPE =================
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

# ================= 2. OUTILS =================
def rotate_if_landscape(img_np):
    if img_np.shape[1] > img_np.shape[0]:
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
    return img_np

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]], dtype=float)
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]], dtype=float)
    dot = float(np.dot(v1, v2))
    mag = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(dot / mag, -1, 1)))

def femur_tibia_knee_angle(hip, knee, ankle):
    return calculate_angle(hip, knee, ankle)

def tibia_rearfoot_ankle_angle(knee, ankle, heel):
    return calculate_angle(knee, ankle, heel)

def safe_point(lm, landmark_enum, w, h):
    p = lm[landmark_enum.value]
    return np.array([p.x * w, p.y * h], dtype=np.float32), float(p.visibility)

def pdf_safe(text) -> str:
    if text is None:
        return ""
    s = str(text)
    s = (s.replace("°", " deg")
           .replace("–", "-")
           .replace("—", "-")
           .replace("’", "'")
           .replace("“", '"')
           .replace("”", '"'))
    return s.encode("latin-1", errors="ignore").decode("latin-1")

def generate_pdf(data, img_np):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(31, 73, 125)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 20, pdf_safe("BILAN POSTURAL IA"), ln=True, align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(25)
    pdf.cell(100, 10, pdf_safe(f"Patient : {data.get('Nom','')}"), ln=0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(90, 10, pdf_safe(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}"), ln=1, align="R")
    pdf.line(10, 68, 200, 68)
    pdf.ln(5)

    img_pil = Image.fromarray(img_np)
    tmp_img = os.path.join(tempfile.gettempdir(), "temp_analysis.png")
    img_pil.save(tmp_img)
    pdf.image(tmp_img, x=60, w=90)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(110, 10, pdf_safe("Indicateur de Mesure"), 1, 0, 'L', True)
    pdf.cell(80, 10, pdf_safe("Valeur"), 1, 1, 'C', True)

    pdf.set_font("Arial", '', 11)
    for k, v in data.items():
        if k != "Nom":
            pdf.cell(110, 9, " " + pdf_safe(k), 1, 0, 'L')
            pdf.cell(80, 9, " " + pdf_safe(v), 1, 1, 'C')

    pdf.set_y(-25)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, pdf_safe("Document généré par Analyseur Postural Pro - Usage indicatif uniquement."), align="C")

    filename = f"Bilan_{pdf_safe(data.get('Nom','Anonyme')).replace(' ', '_')}.pdf"
    pdf.output(filename)

    if os.path.exists(tmp_img):
        os.remove(tmp_img)

    return filename

# ================= 3. SESSION =================
if "override_one" not in st.session_state:
    st.session_state["override_one"] = {}  # {"Cheville G": (x,y)}

# ================= 4. UI =================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille_cm = st.number_input("Taille (cm)", min_value=100, max_value=220, value=170)
    vue = st.selectbox("Vue de la photo", ["Face", "Dos"], index=0)

    st.divider()
    source = st.radio("Source de l'image", ["📷 Caméra", "📁 Téléverser une photo"])

    st.divider()
    st.subheader("🖱️ Correction avant analyse")
    enable_click_edit = st.checkbox("Activer correction par clic", value=True)

    editable_points = ["Genou G", "Genou D", "Cheville G", "Cheville D", "Talon G", "Talon D"]
    point_to_edit = st.selectbox("Point à corriger", editable_points, disabled=not enable_click_edit)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Reset point", disabled=not enable_click_edit):
            st.session_state["override_one"].pop(point_to_edit, None)
    with c2:
        if st.button("🧹 Reset", disabled=not enable_click_edit):
            st.session_state["override_one"] = {}

col_input, col_result = st.columns([1, 1])

# ================= 5. INPUT IMAGE =================
with col_input:
    if source == "📷 Caméra":
        image_data = st.camera_input("Capturez la posture")
    else:
        image_data = st.file_uploader("Format JPG/PNG", type=["jpg", "png", "jpeg"])

if not image_data:
    st.stop()

if isinstance(image_data, Image.Image):
    img_np = np.array(image_data.convert("RGB"))
else:
    img_np = np.array(Image.open(image_data).convert("RGB"))

img_np = rotate_if_landscape(img_np)
h, w, _ = img_np.shape

# ================= 6. IMAGE CLIQUABLE (remplace le canvas) =================
with col_input:
    st.subheader("📌 Cliquez pour placer le point sélectionné (avant analyse)")
    st.caption("Choisis un point à gauche, clique sur l'image, puis lance l'analyse.")

    # On affiche une version redimensionnée pour clic confortable
    disp_w = min(900, w)
    scale = disp_w / w
    disp_h = int(h * scale)

    img_disp = cv2.resize(img_np, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    coords = streamlit_image_coordinates(
        Image.fromarray(img_disp),
        key="img_click",
    )

    if enable_click_edit and coords is not None:
        # coords en pixels sur image affichée
        cx = float(coords["x"])
        cy = float(coords["y"])

        # Convertir vers image originale
        x_orig = cx / scale
        y_orig = cy / scale

        st.session_state["override_one"][point_to_edit] = (x_orig, y_orig)
        st.success(f"✅ {point_to_edit} placé à ({x_orig:.0f}, {y_orig:.0f}) px")

    if st.session_state["override_one"]:
        st.write("**Point corrigé :**")
        for k, (x, y) in st.session_state["override_one"].items():
            st.write(f"- {k} → ({x:.0f}, {y:.0f})")

# ================= 7. ANALYSE =================
with col_result:
    st.subheader("⚙️ Analyse")
    run = st.button("▶ Lancer l'analyse", use_container_width=True)

if not run:
    st.stop()

with st.spinner("Détection (MediaPipe) + calculs..."):
    res = pose.process(img_np)  # img_np est RGB uint8
    if not res.pose_landmarks:
        st.error("Aucune pose détectée. Photo plus nette, en pied, bien centrée.")
        st.stop()

    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    LS, _ = safe_point(lm, L.LEFT_SHOULDER, w, h)
    RS, _ = safe_point(lm, L.RIGHT_SHOULDER, w, h)
    LH, _ = safe_point(lm, L.LEFT_HIP, w, h)
    RH, _ = safe_point(lm, L.RIGHT_HIP, w, h)
    LK, _ = safe_point(lm, L.LEFT_KNEE, w, h)
    RK, _ = safe_point(lm, L.RIGHT_KNEE, w, h)
    LA, _ = safe_point(lm, L.LEFT_ANKLE, w, h)
    RA, _ = safe_point(lm, L.RIGHT_ANKLE, w, h)
    LHE, _ = safe_point(lm, L.LEFT_HEEL, w, h)
    RHE, _ = safe_point(lm, L.RIGHT_HEEL, w, h)

    POINTS = {
        "Epaule G": LS, "Epaule D": RS,
        "Hanche G": LH, "Hanche D": RH,
        "Genou G": LK, "Genou D": RK,
        "Cheville G": LA, "Cheville D": RA,
        "Talon G": LHE, "Talon D": RHE,
    }

    # Appliquer correction (1 point)
    for k, (x, y) in st.session_state["override_one"].items():
        if k in POINTS:
            POINTS[k] = np.array([x, y], dtype=np.float32)

    LS = POINTS["Epaule G"]; RS = POINTS["Epaule D"]
    LH = POINTS["Hanche G"]; RH = POINTS["Hanche D"]
    LK = POINTS["Genou G"];  RK = POINTS["Genou D"]
    LA = POINTS["Cheville G"]; RA = POINTS["Cheville D"]
    LHE = POINTS["Talon G"]; RHE = POINTS["Talon D"]

    raw_sh = math.degrees(math.atan2(LS[1]-RS[1], LS[0]-RS[0]))
    shoulder_angle = abs(raw_sh)
    if shoulder_angle > 90:
        shoulder_angle = abs(shoulder_angle - 180)

    raw_hip = math.degrees(math.atan2(LH[1]-RH[1], LH[0]-RH[0]))
    hip_angle = abs(raw_hip)
    if hip_angle > 90:
        hip_angle = abs(hip_angle - 180)

    knee_l = femur_tibia_knee_angle(LH, LK, LA)
    knee_r = femur_tibia_knee_angle(RH, RK, RA)
    ankle_l = tibia_rearfoot_ankle_angle(LK, LA, LHE)
    ankle_r = tibia_rearfoot_ankle_angle(RK, RA, RHE)

    px_height = max(LA[1], RA[1]) - min(LS[1], RS[1])
    mm_per_px = (float(taille_cm) * 10.0) / px_height if px_height > 0 else 0.0
    diff_shoulders_mm = abs(LS[1] - RS[1]) * mm_per_px
    diff_hips_mm = abs(LH[1] - RH[1]) * mm_per_px

    shoulder_lower = "Gauche" if LS[1] > RS[1] else "Droite"
    hip_lower = "Gauche" if LH[1] > RH[1] else "Droite"
    if vue == "Dos":
        shoulder_lower = "Droite" if shoulder_lower == "Gauche" else "Gauche"
        hip_lower = "Droite" if hip_lower == "Gauche" else "Gauche"

    annotated = img_np.copy()
    for _, p in POINTS.items():
        cv2.circle(annotated, tuple(p.astype(int)), 7, (0, 255, 0), -1)

    for name in list(st.session_state["override_one"].keys()):
        if name in POINTS:
            p = POINTS[name]
            cv2.circle(annotated, tuple(p.astype(int)), 14, (255, 0, 255), 3)

    cv2.line(annotated, tuple(LS.astype(int)), tuple(RS.astype(int)), (255, 0, 0), 3)
    cv2.line(annotated, tuple(LH.astype(int)), tuple(RH.astype(int)), (255, 0, 0), 3)

    results = {
        "Nom": nom,
        "Vue (choisie)": vue,
        "Inclinaison Epaules (horizon=0)": f"{shoulder_angle:.1f} deg",
        "Epaule la plus basse": shoulder_lower,
        "Denivele Epaules (mm)": f"{diff_shoulders_mm:.1f} mm",
        "Inclinaison Bassin (horizon=0)": f"{hip_angle:.1f} deg",
        "Bassin le plus bas": hip_lower,
        "Denivele Bassin (mm)": f"{diff_hips_mm:.1f} mm",
        "Angle Genou Gauche (femur-tibia)": f"{knee_l:.1f} deg",
        "Angle Genou Droit (femur-tibia)": f"{knee_r:.1f} deg",
        "Cheville G (tibia-arriere-pied)": f"{ankle_l:.1f} deg",
        "Cheville D (tibia-arriere-pied)": f"{ankle_r:.1f} deg",
    }

with col_result:
    st.subheader("📊 Résultats")
    st.table(results)

    st.subheader("🖼️ Image annotée")
    st.image(annotated, caption="Points (vert) + point corrigé (violet)", use_container_width=True)

    st.subheader("📄 PDF")
    pdf_path = generate_pdf(results, annotated)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Télécharger le Bilan PDF",
            data=f,
            file_name=pdf_path,
            mime="application/pdf",
            use_container_width=True
        )
