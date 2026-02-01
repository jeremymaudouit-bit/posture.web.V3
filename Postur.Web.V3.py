import streamlit as st
import numpy as np
import cv2
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub

# ... (Gardez vos fonctions load_movenet et preprocess identiques) ...

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    dot = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    return math.degrees(math.acos(np.clip(dot / mag, -1, 1))) if mag > 0 else 0

def generate_pdf(data, img_np):
    pdf = FPDF()
    pdf.add_page()
    
    # --- En-tête ---
    pdf.set_fill_color(41, 128, 185) # Bleu pro
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 20, "BILAN D'ANALYSE POSTURALE", ln=True, align="C")
    
    # --- Infos Patient ---
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(25)
    pdf.cell(95, 10, f"Patient : {data['Nom']}", ln=0)
    pdf.cell(95, 10, f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1, align="R")
    pdf.line(10, 65, 200, 65)
    pdf.ln(10)

    # --- Photo centrée ---
    img_pil = Image.fromarray(img_np)
    img_path = "temp_image.png"
    img_pil.save(img_path)
    # Calcul pour centrer l'image
    pdf.image(img_path, x=55, w=100)
    pdf.ln(5)

    # --- Tableau des résultats ---
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(100, 10, "Mesure", 1, 0, 'C', True)
    pdf.cell(90, 10, "Valeur", 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 11)
    for k, v in data.items():
        if k not in ['Nom']:
            pdf.cell(100, 9, f" {k}", 1, 0, 'L')
            # Coloration si inclinaison importante (optionnel)
            pdf.cell(90, 9, f" {v}", 1, 1, 'C')

    # --- Note de bas de page ---
    pdf.set_y(-30)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, "Note : Ce rapport est généré par IA à titre indicatif. \nUne consultation chez un professionnel de santé est recommandée pour un diagnostic clinique.", align="C")

    filename = f"Bilan_{data['Nom'].replace(' ', '_')}.pdf"
    pdf.output(filename)
    return filename

# ================= ANALYSE (Modifiée pour les angles à 0°) =================
# ... (Dans votre bloc if image_data et bouton analyse) ...

            # Calcul des angles par rapport à l'horizontale (0°)
            # atan2(y2-y1, x2-x1) donne l'angle du vecteur. 
            # On utilise abs() car peu importe le côté du basculement pour le bilan global
            raw_shoulder_angle = math.degrees(math.atan2(LS[1]-RS[1], LS[0]-RS[0]))
            shoulder_angle = abs(raw_shoulder_angle)
            if shoulder_angle > 90: shoulder_angle = abs(shoulder_angle - 180)

            raw_hip_angle = math.degrees(math.atan2(LH[1]-RH[1], LH[0]-RH[0]))
            hip_angle = abs(raw_hip_angle)
            if hip_angle > 90: hip_angle = abs(hip_angle - 180)

            # ... (Reste du calcul des mm inchangé) ...

            results = {
                "Nom": nom,
                "Inclinaison Épaules (0°=Horiz.)": f"{shoulder_angle:.1f}°",
                "Inclinaison Bassin (0°=Horiz.)": f"{hip_angle:.1f}°",
                "Déséquilibre Épaules (Hauteur)": f"{diff_shoulders_mm:.1f} mm",
                "Déséquilibre Bassin (Hauteur)": f"{diff_hips_mm:.1f} mm",
                "Flexion Genou Gauche": f"{knee_l:.1f}°",
                "Flexion Genou Droit": f"{knee_r:.1f}°",
                "Angle Tibia/Verticale G.": f"{ankle_l:.1f}°",
                "Angle Tibia/Verticale D.": f"{ankle_r:.1f}°"
            }
