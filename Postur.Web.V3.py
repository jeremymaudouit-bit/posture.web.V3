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
from streamlit_drawable_canvas import st_canvas

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
    # Genou = angle fémur–tibia => Hanche–Genou–Cheville
    return calculate_angle(hip, knee, ankle)

def tibia_rearfoot_ankle_angle(knee, ankle, heel):
    # Cheville = angle tibia – arrière-pied => Genou–Cheville–Talon
    return calculate_angle(knee, ankle, heel)

def safe_point(lm, landmark_enum, w, h):
    p = lm[landmark_enum.value]
    return np.array([p.x * w, p.y * h], dtype=np.float32), float(p.visibility)

# -------- PDF safe (évite Unicode crash) --------
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

# ================= 3. SESSION STATE (points modifiés) =================
# overrides = positions absolues en pixels (x,y) sur image originale
if "overrides" not in st.session_state:
    st.session_state["overrides"] = {}  # ex: {"Cheville G": (x,y)}

# ================= 4. UI =================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille_cm = st.number_input("Taille (cm)", min_value=100, max_value=220, value=170)

    vue = st.selectbox("Vue de la photo", ["Face", "Dos"], index=0)

    st.divider()
    source = st.radio("Source de l'image", ["📷 Caméra", "📁 Téléverser une photo"])

    st.divider()
    st.subheader("🖱️ Correction par clic (1 point)")
    enable_click_edit = st.checkbox("Activer correction par clic", value=False)

    editable_points = ["Genou G", "Genou D", "Cheville G", "Cheville D", "Talon G", "Talon D"]
    point_to_edit = st.selectbox("Point à corriger", editable_points, disabled=not enable_click_edit)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Reset point", disabled=not enable_click_edit):
            st.session_state["overrides"].pop(point_to_edit, None)
    with c2:
        if st.button("🧹 Reset tous", disabled=not enable_click_edit):
            st.session_state["overrides"] = {}

col_input, col_result = st.columns([1, 1])

image_data = None
with col_input:
    if source == "📷 Caméra":
        st.write("La caméra par défaut du navigateur sera utilisée.")
        image_data = st.camera_input("Capturez la posture")
    else:
        image_data = st.file_uploader("Format JPG/PNG", type=["jpg", "png", "jpeg"])

# ================= 5. ANALYSE =================
if image_data:
    if isinstance(image_data, Image.Image):
        img = image_data.convert("RGB")
        img_np = np.array(img)
    else:
        img = Image.open(image_data).convert("RGB")
        img_np = np.array(img)

    img_np = rotate_if_landscape(img_np)
    h, w, _ = img_np.shape

    # Bouton analyse
    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE", use_container_width=True):
        with st.spinner("Détection de la posture (MediaPipe)..."):
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)

            if not res.pose_landmarks:
                st.error("Aucune pose détectée. Photo plus nette, en pied, bien centrée.")
                st.stop()

            lm = res.pose_landmarks.landmark
            L = mp_pose.PoseLandmark

            # Points MediaPipe
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

            # Appliquer overrides (positions absolues)
            for k, xy in st.session_state["overrides"].items():
                if k in POINTS:
                    POINTS[k] = np.array([xy[0], xy[1]], dtype=np.float32)

            # Réassignation
            LS = POINTS["Epaule G"]; RS = POINTS["Epaule D"]
            LH = POINTS["Hanche G"]; RH = POINTS["Hanche D"]
            LK = POINTS["Genou G"];  RK = POINTS["Genou D"]
            LA = POINTS["Cheville G"]; RA = POINTS["Cheville D"]
            LHE = POINTS["Talon G"]; RHE = POINTS["Talon D"]

            # Inclinaison épaules/bassin
            raw_shoulder_angle = math.degrees(math.atan2(LS[1]-RS[1], LS[0]-RS[0]))
            shoulder_angle = abs(raw_shoulder_angle)
            if shoulder_angle > 90:
                shoulder_angle = abs(shoulder_angle - 180)

            raw_hip_angle = math.degrees(math.atan2(LH[1]-RH[1], LH[0]-RH[0]))
            hip_angle = abs(raw_hip_angle)
            if hip_angle > 90:
                hip_angle = abs(hip_angle - 180)

            # Angles
            knee_l = femur_tibia_knee_angle(LH, LK, LA)
            knee_r = femur_tibia_knee_angle(RH, RK, RA)
            ankle_l = tibia_rearfoot_ankle_angle(LK, LA, LHE)
            ankle_r = tibia_rearfoot_ankle_angle(RK, RA, RHE)

            # Échelle mm/pixel
            px_height = max(LA[1], RA[1]) - min(LS[1], RS[1])
            mm_per_px = (float(taille_cm) * 10.0) / px_height if px_height > 0 else 0.0
            diff_shoulders_mm = abs(LS[1] - RS[1]) * mm_per_px
            diff_hips_mm = abs(LH[1] - RH[1]) * mm_per_px

            shoulder_lower = "Gauche" if LS[1] > RS[1] else "Droite"
            hip_lower = "Gauche" if LH[1] > RH[1] else "Droite"

            if vue == "Dos":
                shoulder_lower = "Droite" if shoulder_lower == "Gauche" else "Gauche"
                hip_lower = "Droite" if hip_lower == "Gauche" else "Gauche"

            # Annotation
            annotated = img_np.copy()

            # points
            for name, p in POINTS.items():
                cv2.circle(annotated, tuple(p.astype(int)), 7, (0, 255, 0), -1)

            # highlight points modifiés
            for name in st.session_state["overrides"].keys():
                if name in POINTS:
                    p = POINTS[name]
                    cv2.circle(annotated, tuple(p.astype(int)), 14, (255, 0, 255), 3)

            # lignes épaules/bassin
            cv2.line(annotated, tuple(LS.astype(int)), tuple(RS.astype(int)), (255, 0, 0), 3)
            cv2.line(annotated, tuple(LH.astype(int)), tuple(RH.astype(int)), (255, 0, 0), 3)

            # segments genou/cheville
            cv2.line(annotated, tuple(LH.astype(int)), tuple(LK.astype(int)), (0, 255, 255), 2)
            cv2.line(annotated, tuple(LK.astype(int)), tuple(LA.astype(int)), (0, 255, 255), 2)
            cv2.line(annotated, tuple(LA.astype(int)), tuple(LHE.astype(int)), (0, 255, 255), 2)

            cv2.line(annotated, tuple(RH.astype(int)), tuple(RK.astype(int)), (0, 255, 255), 2)
            cv2.line(annotated, tuple(RK.astype(int)), tuple(RA.astype(int)), (0, 255, 255), 2)
            cv2.line(annotated, tuple(RA.astype(int)), tuple(RHE.astype(int)), (0, 255, 255), 2)

            cv2.putText(annotated, f"Vue: {vue}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(annotated, f"Epaules: {shoulder_lower} plus basse ({diff_shoulders_mm:.1f} mm)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(annotated, f"Bassin: {hip_lower} plus bas ({diff_hips_mm:.1f} mm)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            # Sauver pour édition clic + affichage
            st.session_state["last_annotated"] = annotated
            st.session_state["last_points"] = {k: (float(v[0]), float(v[1])) for k, v in POINTS.items()}
            st.session_state["last_results"] = {
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

    # ================= 6. AFFICHAGE + CANVAS CLIC =================
    if "last_annotated" in st.session_state:
        annotated = st.session_state["last_annotated"]
        results = st.session_state["last_results"]
        last_points = st.session_state["last_points"]

        with col_result:
            st.subheader("Résultats")
            st.table(results)

        # Canvas dans la colonne input (plus logique : tu cliques sur l'image)
        with col_input:
            st.subheader("Image (clic pour corriger un point)")
            st.caption("Active la correction, choisis un point, puis clique sur l'image.")

            # Affichage à une largeur raisonnable
            disp_w = min(850, annotated.shape[1])
            scale = disp_w / annotated.shape[1]
            disp_h = int(annotated.shape[0] * scale)

            # Image à afficher dans le canvas (RGB)
            img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

            canvas = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=2,
                stroke_color="#ff00ff",
                background_image=Image.fromarray(img_rgb),
                update_streamlit=True,
                height=disp_h,
                width=disp_w,
                drawing_mode="point" if enable_click_edit else "transform",
                key="canvas_edit",
            )

            if enable_click_edit and canvas.json_data is not None:
                objs = canvas.json_data.get("objects", [])
                if len(objs) > 0:
                    # Dernier point cliqué
                    obj = objs[-1]
                    cx = obj.get("left", None)
                    cy = obj.get("top", None)

                    if cx is not None and cy is not None:
                        # Convertir coords canvas -> coords image originale
                        x_orig = float(cx) / scale
                        y_orig = float(cy) / scale

                        # Appliquer sur le point choisi
                        st.session_state["overrides"][point_to_edit] = (x_orig, y_orig)
                        st.success(f"✅ {point_to_edit} déplacé à ({x_orig:.0f}, {y_orig:.0f}) px. Relance l'analyse.")

                        # Nettoyage des clicks (sinon ça s'accumule)
                        # Trick simple : changer la key du canvas au prochain run en fonction d'un compteur
                        st.session_state["canvas_bump"] = st.session_state.get("canvas_bump", 0) + 1

        # ================= 7. PDF =================
        with col_result:
            st.subheader("PDF")
            pdf_path = generate_pdf(results, annotated)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Télécharger le Bilan PDF",
                    data=f,
                    file_name=pdf_path,
                    mime="application/pdf",
                    use_container_width=True
                )
