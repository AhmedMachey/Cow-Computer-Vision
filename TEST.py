import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# PARAMÈTRES
# ===============================
image_path = r"C:\Users\AHMED\OneDrive\Desktop\projet\src\vache1.jpg"
COEFF_A = 0.0025
COEFF_B = 50.0
POIDS_SEUIL = 650  # kg

# ===============================
# FONCTION : SEGMENTATION IMAGE
# ===============================
def segmenter_vache_image(image_path, a, b):
    img = cv2.imread(image_path)
    if img is None:
        print("Image introuvable")
        return None, None, 0, "Inconnu"

    h, w = img.shape[:2]

    # GrabCut réduit (3-5 itérations suffisent)
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, w - 20, h - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask_bin = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

    # Morphologie
    kernel = np.ones((15, 15), np.uint8)
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=2)

    # Contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    if not contours:
        return img, mask_clean, 0, "Inconnu"

    largest = max(contours, key=cv2.contourArea)
    poids = a * cv2.contourArea(largest) + b

    # Classification
    classification = "Vache de viande" if poids > POIDS_SEUIL else "Vache laitiere"

    # Dessin sur l'image
    out = img.copy()
    x, y, w2, h2 = cv2.boundingRect(largest)
    cv2.drawContours(out, [largest], -1, (0, 128, 255), 3)  # contours orange
    cv2.rectangle(out, (x, y), (x + w2, y + h2), (255, 0, 0), 2)  # bounding box bleue
    cv2.putText(out, f"Poids : {poids:.1f} kg", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(out, f"Type : {classification}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    return out, mask_clean, poids, classification

# ===============================
# EXÉCUTION IMAGE
# ===============================
result, mask, poids, classification = segmenter_vache_image(image_path, COEFF_A, COEFF_B)

# ===============================
# AFFICHAGE PROFESSIONNEL
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle(f"Vache détectée : {classification} - Poids estimé = {poids:.1f} kg",
             fontsize=16, fontweight='bold')

axes[0].imshow(mask, cmap='gray')
axes[0].set_title("Masque segmenté", fontsize=14)
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[1].set_title("Contour et Poids", fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.show()
