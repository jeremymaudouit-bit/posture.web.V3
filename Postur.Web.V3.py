# ==============================
# IMPORTS
# ==============================
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as PDFImage,
    Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

# ==============================
# CONFIG
# ==============================
st.set_page_config("GaitScan Pro", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique")
FPS = 30

# ==============================
# MOVENET
# ==============================
@st.cache_resource
def load_movenet():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

movenet = load_movenet()

def detect_pose(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(img[None], 192, 192)
    out = movenet.signatures["serving_default"](tf.cast(img, tf.int32))
    return out["output_0"].numpy()[0, 0]  # (17, 3): y, x, score

# ==============================
# JOINTS
# ==============================
J = {
    "Epaule G": 5,  "Epaule D": 6,
    "Hanche G": 11, "Hanche D": 12,
    "Genou G": 13,  "Genou D": 14,
    "Cheville G": 15, "Cheville D": 16
}

LEFT_RIGHT_PAIRS = [
    ("Epaule G", "Epaule D"),
    ("Hanche G", "Hanche D"),
    ("Genou G", "Genou D"),
    ("Cheville G", "Cheville D"),
]

# ==============================
# ANGLES
# ==============================
def angle_genou(a, b, c):
    """
    Angle du genou (0 = plié, 180 = droit)
    Points: a (hanche), b (genou), c (cheville)
    """
    ba = a - b
    bc = c - b
    # inversion axe Y pour repère "classique" (y vers le haut)
    ba[1] *= -1
    bc[1] *= -1

    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    ang = np.clip(np.degrees(np.arccos(cos_theta)), 0, 180)
    return ang

def angle(a, b, c, joint_type="Hanche/Cheville"):
    """
    Autres articulations : hanche ou cheville
    """
    ba = a - b
    bc = c - b
    ba[1] *= -1
    bc[1] *= -1

    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    ang = np.clip(np.degrees(np.arccos(cos_theta)), 0, 180)

    if joint_type == "Hanche":
        return 180 - ang
    elif joint_type == "Cheville":
        return 90 - (ang - 90)
    else:
        return ang

def wrap_deg_180(deg):
    """Normalise un angle en degrés dans [-180, 180)."""
    return (deg + 180) % 360 - 180

# ==============================
# BANDPASS
# ==============================
def bandpass(sig, level, fs=FPS):
    low = 0.3 + level * 0.02
    high = max(6.0 - level * 0.25, low + 0.4)
    b, a = butter(2, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)

# ==============================
# VIDEO PROCESS
# ==============================
def process_video(path, mirror_lr=False):
    """
    mirror_lr=True => la vidéo est "miroir" (selfie/preview) : on inverse anatomiquement gauche/droite
    """
    cap = cv2.VideoCapture(path)

    res = {
        "Hanche G": [], "Hanche D": [],
        "Genou G": [],  "Genou D": [],
        "Cheville G": [], "Cheville D": [],
        "Pelvis": [], "Dos": []
    }
    heel_y_D, frames = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        kp = detect_pose(frame)  # y,x,score (0..1)
        frames.append(frame.copy())

        # Option miroir: swap des keypoints anatomiques gauche/droite
        if mirror_lr:
            kp = kp.copy()
            for L, R in LEFT_RIGHT_PAIRS:
                iL, iR = J[L], J[R]
                kp[[iL, iR]] = kp[[iR, iL]]

        # ANGLES (on travaille en coordonnées (x,y) normalisées)
        # kp[:, :2] = (y, x) -> on veut (x, y) pour nos fonctions
        def xy(idx):
            yx = kp[idx, :2]
            return np.array([yx[1], yx[0]], dtype=np.float32)  # (x, y)

        # Hanches
        res["Hanche G"].append(angle(xy(J["Epaule G"]), xy(J["Hanche G"]), xy(J["Genou G"]), "Hanche"))
        res["Hanche D"].append(angle(xy(J["Epaule D"]), xy(J["Hanche D"]), xy(J["Genou D"]), "Hanche"))

        # Genoux
        res["Genou G"].append(angle_genou(xy(J["Hanche G"]), xy(J["Genou G"]), xy(J["Cheville G"])))
        res["Genou D"].append(angle_genou(xy(J["Hanche D"]), xy(J["Genou D"]), xy(J["Cheville D"])))

        # Chevilles (référence verticale locale)
        res["Cheville G"].append(angle(xy(J["Genou G"]), xy(J["Cheville G"]), xy(J["Cheville G"]) + np.array([0, 1], np.float32), "Cheville"))
        res["Cheville D"].append(angle(xy(J["Genou D"]), xy(J["Cheville D"]), xy(J["Cheville D"]) + np.array([0, 1], np.float32), "Cheville"))

        # PELVIS (orientation du segment hancheG -> hancheD)
        pelvis = xy(J["Hanche D"]) - xy(J["Hanche G"])
        # inversion Y (image y vers le bas) pour angle "classique"
        pelvis[1] *= -1
        pel_ang = np.degrees(np.arctan2(pelvis[1], pelvis[0]))  # [-180, 180]
        pel_ang = wrap_deg_180(pel_ang)  # évite 358° (= -2°)
        res["Pelvis"].append(pel_ang)

        # DOS (inclinaison tronc)
        mid_hip = (xy(11) + xy(12)) / 2
        mid_sh = (xy(5) + xy(6)) / 2
        res["Dos"].append(angle(mid_sh, mid_hip, mid_hip + np.array([0, -1], np.float32)))

        heel_y_D.append(kp[J["Cheville D"], 0])  # y normalisé

    cap.release()

    # Optionnel mais recommandé: unwrap pelvis pour éviter des sauts +/-180 dans le signal
    pel = np.array(res["Pelvis"], dtype=float)
    pel_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(pel)))
    res["Pelvis"] = pel_unwrapped.tolist()

    return res, heel_y_D, frames

# ==============================
# CYCLE DETECTION
# ==============================
def detect_cycle(heel_y):
    inv = -np.array(heel_y)
    peaks, _ = find_peaks(inv, distance=FPS // 2, prominence=np.std(inv) * 0.3)
    if len(peaks) >= 2:
        return peaks[0], peaks[1]
    return 0, len(heel_y) - 1

# ==============================
# NORMES
# ==============================
def norm_curve(joint, n):
    x = np.linspace(0, 100, n)
    if joint == "Genou":
        return np.interp(x, [0, 15, 40, 60, 80, 100], [180, 170, 180, 140, 120, 180])
    if joint == "Hanche":
        return np.interp(x, [0, 30, 60, 100], [180, 160, 150, 180])
    if joint == "Cheville":
        return np.interp(x, [0, 10, 50, 70, 100], [90, 85, 100, 75, 90])
    return np.zeros(n)

# ==============================
# PDF EXPORT
# ==============================
def export_pdf(patient, keyframe, figures, table_data):
    path = os.path.join(tempfile.gettempdir(), "rapport_gaitscan.pdf")
    doc = SimpleDocTemplate(
        path, pagesize=A4, rightMargin=2 * cm,
        leftMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>GaitScan Pro – Analyse Cinématique</b>", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"<b>Patient :</b> {patient['nom']} {patient['prenom']}<br/>"
        f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y')}<br/>"
        f"<b>Caméra :</b> {patient.get('camera', 'N/A')}<br/>"
        f"<b>Vidéo miroir :</b> {'Oui' if patient.get('mirror', False) else 'Non'}",
        styles["Normal"]
    ))
    story.append(Paragraph(
        f"<b>Phase du pas basée sur :</b> {patient.get('phase_cote', 'N/A')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("<b>Image représentative du cycle</b>", styles["Heading2"]))
    story.append(PDFImage(keyframe, width=16 * cm, height=8 * cm))
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("<b>Analyse articulaire</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.3 * cm))
    for joint, img_path in figures.items():
        story.append(Paragraph(f"<b>{joint}</b>", styles["Heading3"]))
        story.append(PDFImage(img_path, width=16 * cm, height=6 * cm))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("<b>Synthèse des angles (°) – Gauche / Droite</b>", styles["Heading2"]))
    table = Table([["Articulation", "Min", "Moyenne", "Max"]] + table_data,
                  colWidths=[5 * cm, 3 * cm, 3 * cm, 3 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER")
    ]))
    story.append(table)
    doc.build(story)
    return path

# ==============================
# STREAMLIT INTERFACE
# ==============================
with st.sidebar:
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    smooth = st.slider("Lissage band-pass", 0, 10, 3)
    src = st.radio("Source", ["Vidéo", "Caméra"])
    camera_pos = st.selectbox("Position de la caméra", ["Devant", "Droite", "Gauche"])
    phase_cote = st.selectbox("Phase du pas basée sur :", ["Droite", "Gauche", "Les deux"])

    # IMPORTANT: pour les vidéos “miroir” (souvent caméra frontale / preview)
    mirror_lr = st.checkbox(
        "Vidéo miroir (inversion gauche/droite)",
        value=False,
        help="Coche si la vidéo est inversée comme un selfie (gauche/droite permutés)."
    )

video = st.file_uploader("Vidéo", ["mp4", "avi", "mov"]) if src == "Vidéo" else st.camera_input("Caméra")

# ==============================
# ANALYSE
# ==============================
if video and st.button("▶ Lancer l'analyse"):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read())
    tmp.close()

    data, heel_y, frames = process_video(tmp.name, mirror_lr=mirror_lr)
    os.unlink(tmp.name)

    # Détecter la phase selon le côté choisi (sur cheville filtrée)
    if phase_cote == "Droite":
        heel_f_ref = bandpass(np.array(data["Cheville D"]), smooth)
        c0, c1 = detect_cycle(heel_f_ref)
        phase_colors = [(c0, c1, "blue")]
    elif phase_cote == "Gauche":
        heel_f_ref = bandpass(np.array(data["Cheville G"]), smooth)
        c0, c1 = detect_cycle(heel_f_ref)
        phase_colors = [(c0, c1, "orange")]
    else:  # Les deux
        heel_f_D = bandpass(np.array(data["Cheville D"]), smooth)
        heel_f_G = bandpass(np.array(data["Cheville G"]), smooth)
        c0_D, c1_D = detect_cycle(heel_f_D)
        c0_G, c1_G = detect_cycle(heel_f_G)
        phase_colors = [(c0_D, c1_D, "blue"), (c0_G, c1_G, "orange")]

    key_img = os.path.join(tempfile.gettempdir(), "keyframe.png")
    cv2.imwrite(key_img, frames[len(frames) // 2])

    figs, table_data = {}, []

    for joint in ["Hanche", "Genou", "Cheville"]:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 4),
            gridspec_kw={"width_ratios": [2, 1]}
        )
        g = bandpass(np.array(data[f"{joint} G"]), smooth)
        d = bandpass(np.array(data[f"{joint} D"]), smooth)
        n = norm_curve(joint, len(g))

        ax1.plot(g, label="Gauche", color="red")
        ax1.plot(d, label="Droite", color="blue")
        for c0, c1, color in phase_colors:
            ax1.axvspan(c0, c1, color=color, alpha=0.3)
        ax1.set_title(f"{joint} – Analyse")
        ax1.legend()

        ax2.plot(n, color="green")
        ax2.set_title("Norme")

        st.pyplot(fig)
        img = os.path.join(tempfile.gettempdir(), f"{joint}.png")
        fig.savefig(img, bbox_inches="tight")
        plt.close(fig)
        figs[joint] = img

        table_data.append([joint + " Gauche", f"{g.min():.1f}", f"{g.mean():.1f}", f"{g.max():.1f}"])
        table_data.append([joint + " Droite", f"{d.min():.1f}", f"{d.mean():.1f}", f"{d.max():.1f}"])

    # PDF
    pdf_path = export_pdf(
        patient={
            "nom": nom,
            "prenom": prenom,
            "camera": camera_pos,
            "phase_cote": phase_cote,
            "mirror": mirror_lr,
        },
        keyframe=key_img,
        figures=figs,
        table_data=table_data
    )

    with open(pdf_path, "rb") as f:
        st.download_button(
            "📄 Télécharger le rapport PDF",
            f,
            file_name=f"GaitScan_{nom}_{prenom}.pdf",
            mime="application/pdf"
        )
