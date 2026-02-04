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
def rotate_if_landscape(img_np_rgb: np.ndarray) -> np.ndarray:
    if img_np_rgb.shape[1] > img_np_rgb.shape[0]:
        img_np_rgb = cv2.rotate(img_np_rgb, cv2.ROTATE_90_CLOCKWISE)
    return img_np_rgb

def ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Force image RGB uint8 contiguë."""
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    return img

def to_png_bytes(img_rgb_uint8: np.ndarray) -> bytes:
    """Encode en PNG bytes (robuste)."""
    img_rgb_uint8 = ensure_uint8_rgb(img_rgb_uint8)
    pil = Image.fromarray(img_rgb_uint8, mode="RGB")
    bio = io.BytesIO()
    pil.save(bio, format="PNG")
    return bio.getvalue()

def calculate_angle(p1, p2, p3) -> float:
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]], dtype=float)
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]], dtype=float)
    dot = float(np.dot(v1, v2))
    mag = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(dot / mag, -1, 1)))

def femur_tibia_knee_angle(hip, knee, ankle) -> float:
    return calculate_angle(hip, knee, ankle)

def tibia_rearfoot_ankle_angle(knee, ankle, heel) -> float:
    return calculate_angle(knee, ankle, heel)

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

def crop_to_landmarks(img_rgb_uint8: np.ndarray, res_pose, pad_ratio: float = 0.18) -> np.ndarray:
    """
    Cadrage auto autour du corps à partir des landmarks MediaPipe.
    pad_ratio = marge relative autour de la bbox.
    """
    if res_pose is None or not res_pose.pose_landmarks:
        return img_rgb_uint8

    h, w = img_rgb_uint8.shape[:2]
    xs, ys = [], []
    for lm in res_pose.pose_landmarks.landmark:
        if lm.visibility < 0.2:
            continue
        xs.append(lm.x * w)
        ys.append(lm.y * h)

    if not xs or not ys:
        return img_rgb_uint8

    x1, x2 = max(0, int(min(xs))), min(w-1, int(max(xs)))
    y1, y2 = max(0, int(min(ys))), min(h-1, int(max(ys)))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_x)
    x2 = min(w-1, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h-1, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return img_rgb_uint8

    return img_rgb_uint8[y1:y2, x1:x2].copy()

# =========================
# PDF PRO (EN MÉMOIRE + COMPAT FPDF)
# =========================
def generate_pdf(data: dict, img_rgb_uint8: np.ndarray) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(31, 73, 125)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(0, 20, pdf_safe("BILAN POSTURAL IA"), ln=True, align="C")

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(25)
    pdf.cell(110, 10, pdf_safe(f"Patient : {data.get('Nom','')}"), ln=0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(80, 10, pdf_safe(datetime.now().strftime('%d/%m/%Y %H:%M')), ln=1, align="R")
    pdf.line(10, 68, 200, 68)
    pdf.ln(5)

    img_rgb_uint8 = ensure_uint8_rgb(img_rgb_uint8)
    tmp_img = os.path.join(tempfile.gettempdir(), "temp_analysis.png")
    Image.fromarray(img_rgb_uint8, mode="RGB").save(tmp_img)
    pdf.image(tmp_img, x=55, w=100)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(120, 10, pdf_safe("Indicateur"), 1, 0, 'L', True)
    pdf.cell(70, 10, pdf_safe("Valeur"), 1, 1, 'C', True)

    pdf.set_font("Arial", '', 11)
    for k, v in data.items():
        if k != "Nom":
            pdf.cell(120, 9, " " + pdf_safe(k), 1, 0, 'L')
            pdf.cell(70, 9, " " + pdf_safe(v), 1, 1, 'C')

    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, pdf_safe("Document indicatif - Ne remplace pas un avis médical."), align="C")

    if os.path.exists(tmp_img):
        os.remove(tmp_img)

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1")

def extract_origin_points_from_mediapipe(img_rgb_uint8: np.ndarray):
    res = pose.process(img_rgb_uint8)
    if not res.pose_landmarks:
        return {}
    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark
    h, w = img_rgb_uint8.shape[:2]

    def pt_px(enum_):
        p = lm[enum_.value]
        return (float(p.x * w), float(p.y * h))

    # ✅ hanches exposées + épaules gardées en "_" (non éditables mais dispo si besoin)
    return {
        "Genou G": pt_px(L.LEFT_KNEE),
        "Genou D": pt_px(L.RIGHT_KNEE),
        "Cheville G": pt_px(L.LEFT_ANKLE),
        "Cheville D": pt_px(L.RIGHT_ANKLE),
        "Talon G": pt_px(L.LEFT_HEEL),
        "Talon D": pt_px(L.RIGHT_HEEL),

        "Hanche G": pt_px(L.LEFT_HIP),
        "Hanche D": pt_px(L.RIGHT_HIP),

        "_Epaule G": pt_px(L.LEFT_SHOULDER),
        "_Epaule D": pt_px(L.RIGHT_SHOULDER),
    }

def draw_preview(img_disp_rgb_uint8: np.ndarray, origin_points: dict, override_one: dict, scale: float) -> np.ndarray:
    out = img_disp_rgb_uint8.copy()
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    for name, p in origin_points.items():
        if name.startswith("_"):
            continue
        x = int(p[0] * scale)
        y = int(p[1] * scale)
        cv2.circle(out_bgr, (x, y), 6, (0, 255, 0), -1)

    for name, p in override_one.items():
        x = int(p[0] * scale)
        y = int(p[1] * scale)
        cv2.circle(out_bgr, (x, y), 10, (255, 0, 255), 3)
        cv2.line(out_bgr, (x - 12, y), (x + 12, y), (255, 0, 255), 2)
        cv2.line(out_bgr, (x, y - 12), (x, y + 12), (255, 0, 255), 2)
        cv2.putText(out_bgr, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

# =========================
# 3) SESSION STATE
# =========================
if "override_one" not in st.session_state:
    st.session_state["override_one"] = {}  # {"Cheville G": (x,y)}

# =========================
# 4) UI
# =========================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille_cm = st.number_input("Taille (cm)", min_value=100, max_value=220, value=170)

    st.divider()
    source = st.radio("Source de l'image", ["📷 Caméra", "📁 Téléverser une photo"])

    st.divider()
    st.subheader("🖱️ Correction avant analyse")
    enable_click_edit = st.checkbox("Activer correction par clic", value=True)

    # ✅ Ajout des hanches
    editable_points = ["Hanche G", "Hanche D", "Genou G", "Genou D", "Cheville G", "Cheville D", "Talon G", "Talon D"]
    point_to_edit = st.selectbox("Point à corriger", editable_points, disabled=not enable_click_edit)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩️ Reset point", disabled=not enable_click_edit):
            st.session_state["override_one"].pop(point_to_edit, None)
    with c2:
        if st.button("🧹 Reset tout", disabled=not enable_click_edit):
            st.session_state["override_one"] = {}

    st.divider()
    st.subheader("🖼️ Affichage")
    disp_w_user = st.slider("Largeur d'affichage (px)", min_value=320, max_value=900, value=520, step=10)
    auto_crop = st.checkbox("Cadrage automatique (autour du corps)", value=True)

col_input, col_result = st.columns([1, 1])

# =========================
# 5) INPUT IMAGE
# =========================
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
img_np = ensure_uint8_rgb(img_np)

# (option) recadrage auto basé sur landmarks
res_for_crop = pose.process(img_np)
if auto_crop:
    img_np = crop_to_landmarks(img_np, res_for_crop, pad_ratio=0.18)
    img_np = ensure_uint8_rgb(img_np)

h, w = img_np.shape[:2]

# =========================
# 6) PREVIEW CLIQUABLE (UNE SEULE IMAGE)
# =========================
with col_input:
    st.subheader("📌 Cliquez pour placer le point sélectionné (avant analyse)")
    st.caption("Verts = points d'origine | Violet = point corrigé")

    disp_w = min(int(disp_w_user), w)
    scale = disp_w / w
    disp_h = int(h * scale)

    img_disp = cv2.resize(img_np, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    img_disp = ensure_uint8_rgb(img_disp)

    origin_points = extract_origin_points_from_mediapipe(img_np)
    preview = draw_preview(img_disp, origin_points, st.session_state["override_one"], scale)

    # ⚠️ streamlit_image_coordinates affiche déjà l'image → pas de st.image() ici
    coords = streamlit_image_coordinates(
        Image.open(io.BytesIO(to_png_bytes(preview))),
        key="img_click",
    )

    if enable_click_edit and coords is not None:
        cx = float(coords["x"])
        cy = float(coords["y"])
        x_orig = cx / scale
        y_orig = cy / scale
        st.session_state["override_one"][point_to_edit] = (x_orig, y_orig)
        st.success(f"✅ {point_to_edit} placé à ({x_orig:.0f}, {y_orig:.0f}) px")

    if st.session_state["override_one"]:
        st.write("**Point(s) corrigé(s) enregistré(s) :**")
        for k, (x, y) in st.session_state["override_one"].items():
            st.write(f"- {k} → ({x:.0f}, {y:.0f})")

# =========================
# 7) ANALYSE
# =========================
with col_result:
    st.subheader("⚙️ Analyse")
    run = st.button("▶ Lancer l'analyse")

if not run:
    st.stop()

with st.spinner("Détection (MediaPipe) + calculs..."):
    res = pose.process(img_np)
    if not res.pose_landmarks:
        st.error("Aucune pose détectée. Photo plus nette, en pied, bien centrée.")
        st.stop()

    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(enum_):
        p = lm[enum_.value]
        return np.array([p.x * w, p.y * h], dtype=np.float32)

    LS = pt(L.LEFT_SHOULDER)
    RS = pt(L.RIGHT_SHOULDER)
    LH = pt(L.LEFT_HIP)
    RH = pt(L.RIGHT_HIP)
    LK = pt(L.LEFT_KNEE)
    RK = pt(L.RIGHT_KNEE)
    LA = pt(L.LEFT_ANKLE)
    RA = pt(L.RIGHT_ANKLE)
    LHE = pt(L.LEFT_HEEL)
    RHE = pt(L.RIGHT_HEEL)

    POINTS = {
        "Epaule G": LS, "Epaule D": RS,
        "Hanche G": LH, "Hanche D": RH,
        "Genou G": LK, "Genou D": RK,
        "Cheville G": LA, "Cheville D": RA,
        "Talon G": LHE, "Talon D": RHE,
    }

    # overrides (clics)
    for k, (x, y) in st.session_state["override_one"].items():
        if k in POINTS:
            POINTS[k] = np.array([x, y], dtype=np.float32)

    LS = POINTS["Epaule G"]; RS = POINTS["Epaule D"]
    LH = POINTS["Hanche G"]; RH = POINTS["Hanche D"]
    LK = POINTS["Genou G"];  RK = POINTS["Genou D"]
    LA = POINTS["Cheville G"]; RA = POINTS["Cheville D"]
    LHE = POINTS["Talon G"]; RHE = POINTS["Talon D"]

    # angles épaules / bassin
    raw_sh = math.degrees(math.atan2(LS[1]-RS[1], LS[0]-RS[0]))
    shoulder_angle = abs(raw_sh)
    if shoulder_angle > 90:
        shoulder_angle = abs(shoulder_angle - 180)

    raw_hip = math.degrees(math.atan2(LH[1]-RH[1], LH[0]-RH[0]))
    hip_angle = abs(raw_hip)
    if hip_angle > 90:
        hip_angle = abs(hip_angle - 180)

    # genoux / chevilles
    knee_l = femur_tibia_knee_angle(LH, LK, LA)
    knee_r = femur_tibia_knee_angle(RH, RK, RA)
    ankle_l = tibia_rearfoot_ankle_angle(LK, LA, LHE)
    ankle_r = tibia_rearfoot_ankle_angle(RK, RA, RHE)

    # mm/px depuis taille
    px_height = max(LA[1], RA[1]) - min(LS[1], RS[1])
    mm_per_px = (float(taille_cm) * 10.0) / px_height if px_height > 0 else 0.0
    diff_shoulders_mm = abs(LS[1] - RS[1]) * mm_per_px
    diff_hips_mm = abs(LH[1] - RH[1]) * mm_per_px

    shoulder_lower = "Gauche" if LS[1] > RS[1] else "Droite"
    hip_lower = "Gauche" if LH[1] > RH[1] else "Droite"

    # Annotated image
    ann_bgr = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)

    for _, p in POINTS.items():
        cv2.circle(ann_bgr, tuple(p.astype(int)), 7, (0, 255, 0), -1)

    for name in list(st.session_state["override_one"].keys()):
        if name in POINTS:
            p = POINTS[name]
            cv2.circle(ann_bgr, tuple(p.astype(int)), 14, (255, 0, 255), 3)
            cv2.putText(ann_bgr, name, (int(p[0]) + 10, int(p[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.line(ann_bgr, tuple(LS.astype(int)), tuple(RS.astype(int)), (255, 0, 0), 3)
    cv2.line(ann_bgr, tuple(LH.astype(int)), tuple(RH.astype(int)), (255, 0, 0), 3)

    annotated = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
    annotated = ensure_uint8_rgb(annotated)

    results = {
        "Nom": nom,
        "Inclinaison Épaules (0=horizon)": f"{shoulder_angle:.1f}°",
        "Épaule la plus basse": shoulder_lower,
        "Dénivelé Épaules (mm)": f"{diff_shoulders_mm:.1f} mm",
        "Inclinaison Bassin (0=horizon)": f"{hip_angle:.1f}°",
        "Bassin le plus bas": hip_lower,
        "Dénivelé Bassin (mm)": f"{diff_hips_mm:.1f} mm",
        "Angle Genou Gauche (fémur-tibia)": f"{knee_l:.1f}°",
        "Angle Genou Droit (fémur-tibia)": f"{knee_r:.1f}°",
        "Angle Cheville G (tibia-arrière-pied)": f"{ankle_l:.1f}°",
        "Angle Cheville D (tibia-arrière-pied)": f"{ankle_r:.1f}°",
    }

# =========================
# 8) SORTIE
# =========================
with col_result:
    st.subheader("📊 Résultats")
    st.table(results)

    st.subheader("🖼️ Image annotée")
    # Streamlit ancien -> use_column_width
    # Affichage plus petit + cadré déjà fait en amont
    st.image(
        Image.fromarray(annotated, mode="RGB"),
        caption="Points verts = utilisés | Violet = corrigé",
        use_column_width=True
    )

    st.subheader("📄 PDF")
    pdf_bytes = generate_pdf(results, annotated)
    pdf_name = f"Bilan_{pdf_safe(results.get('Nom','Anonyme')).replace(' ', '_')}.pdf"
    st.download_button(
        label="📥 Télécharger le Bilan PDF",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
    )
