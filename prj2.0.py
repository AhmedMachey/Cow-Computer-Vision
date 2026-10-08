import cv2
import numpy as np

# ===============================
# PARAMÈTRES
# ===============================
video_path = r"C:\Users\AHMED\OneDrive\Desktop\projet\src\Video Project.mp4"
COEFF_A = 0.0006
COEFF_B = 50.0
resize_factor = 0.5  # Réduire la taille pour accélérer le traitement

# ===============================
# INITIALISATION VIDÉO
# ===============================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la vidéo")
    exit()

poids_total = 0
frames_count = 0
mask_prev = None  # Pour suivre la vache d'une frame à l'autre

# ===============================
# TRAITEMENT FRAME PAR FRAME
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionner pour accélérer
    frame_small = cv2.resize(frame, (0,0), fx=resize_factor, fy=resize_factor)
    h, w = frame_small.shape[:2]

    # ----------- GrabCut ----------------
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    if mask_prev is None:
        # Première frame : rectangle large
        rect = (10, 10, w - 20, h - 20)
        cv2.grabCut(frame_small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    else:
        # Frames suivantes : utiliser le masque précédent
        # Convertir mask_prev en valeurs compatibles GrabCut
        mask_gc_for_grabcut = np.where(mask_prev == 0, cv2.GC_BGD, cv2.GC_FGD).astype('uint8')
        cv2.grabCut(frame_small, mask_gc_for_grabcut, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)
        mask = mask_gc_for_grabcut

    # Masque binaire
    mask_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # ----------- Morphologie pour nettoyage ----------------
    kernel_size = max(5, w // 100)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # ----------- Contours ----------------
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 500]

    out = frame_small.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        surface_px = cv2.contourArea(largest)
        x, y, w2, h2 = cv2.boundingRect(largest)
        poids = COEFF_A * surface_px + COEFF_B

        # Dessin
        cv2.drawContours(out, [largest], -1, (0, 0, 255), 2)
        cv2.rectangle(out, (x, y), (x + w2, y + h2), (255, 0, 0), 2)
        cv2.putText(out, f"Poids : {poids:.1f} kg", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mise à jour du poids moyen
        poids_total += poids
        frames_count += 1
        poids_moyen = poids_total / frames_count
        cv2.putText(out, f"Poids moyen : {poids_moyen:.1f} kg", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Mettre à jour mask_prev pour la frame suivante
        mask_prev = mask_clean.copy()
    else:
        cv2.putText(out, "Vache non detectee", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ----------- Affichage ----------------
    cv2.imshow("Vache Segmentée", out)
    cv2.imshow("Masque", mask_clean)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
